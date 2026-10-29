"""Application, parent-comment design (30 Sept 2026): Z = the comment being replied to, rows = the replies to it.
Same corpus and question as app_cmv.py (r/ChangeMyView, ConvoKit winning-args-corpus), but the unit of conditioning is the
parent comment instead of the original post, so that X and Y depend more on Z (the post explains 10% of politeness and 3% of
the score; the parent comment explains about 12% of length and 8% of the score, brainstorm of 30 Sept).

Rows: replies whose parent is a comment (not the post), not written by the original poster, not by a bot, not deleted, not
a reply to one's own comment, at least 20 words, with a score; the parent is not deleted and has at least 5 words; one reply
per user per parent (the earliest); at most MAX_PER_PARENT replies per parent (random, seed 0); parents with at least two
rows are kept. Z = the parent comment text. X: politeness rating (same prompt and model as app_cmv.py), log length, politeness
lexicon, quotes the parent. Y: signed log score, delta awarded to this exact comment, the original poster replied to it.
Views of Z: E1, E2 (parent text), S (surface statistics of the parent text), L300 (TF-IDF singular directions of the parent
text), M (parent metadata: log words, signed log score, written by the original poster, depth, log number of replies,
quote, URL), P1 (E1 of the post), and combinations. Folds grouped by post, standard errors clustered by post, least squares.

Phases (resumable): build -> rows_deep.parquet, parents_deep.parquet (cache ~/.cache/ci_text/cmv), app_cmv_deep_build.json
                    embed -> views_deep.npz
                    rate  -> ratings_<rater>_deep.jsonl (pilot of 20, budget guard); agree -> agreement with the Haiku ratings on the overlap
                    test  -> app_cmv_deep_results.csv, app_cmv_deep_bench.csv
                    components -> app_cmv_deep_components.json
                    plots -> figures app_cmv_deep_main.pdf, app_cmv_deep_views.pdf (PAPER_DIR)
                    macros -> \\resDeep* block in PAPER_DIR/results_macros.tex (--write)
usage: python experiments/app_cmv_deep.py [--only build,embed,rate,agree,test,components,plots,macros] [--rater openai|jev|haiku] [--budget 20] [--write]
       [--regressor ols|mlp] [--smoke]
Regressor (7 Oct 2026): the paper uses least squares. --regressor mlp fits the small networks of citext/nn.py in the test phase (six main
pairs; every embedding for the two score pairs, E1 for the others; the six benchmarks of E1 for every pair; resumable per embedding) and
writes app_cmv_deep_results_<rater>_nn.csv, ..._bench_<rater>_nn.csv, ..._components_<rater>_nn.json, the figures with the suffix _nn and,
with --write, a separate block of \\resDeepNn* macros (not used in the paper) with the least squares values as ...Ls. --smoke: the first pair
with E1, one seed and one configuration, into *_smoke files. Slurm job: app_nn.sbatch (400G: 128 workers each copy the [E1,E2] matrix).
"""
from __future__ import annotations
import argparse, asyncio, html, json, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("CI_TEXT_CODE_DIR", HERE.parent))          # code tree (parent of experiments/), or CI_TEXT_CODE_DIR when run from elsewhere
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "experiments"))
from citext import plotting as P
from citext.estimators import fit_predict_crossfit, partial_r2_benchmark, rv_alpha, critical_value, config_feasible
from citext.nn import crossfit_nn, cfg_summary
from citext.paths import RESULTS_DIR, CACHE_DIR, PAPER_DIR, FIG_DIR
from citext.text_semi import E1_NAME, E2_NAME
from app_cmv import lexicon, embed_len, surface, corr_stats, SYSTEM, PROMPT   # noqa: E402
try:
    from dotenv import load_dotenv; load_dotenv(CODE / ".env")
except Exception:
    pass
CMV = CACHE_DIR / "cmv"; CORPUS = CMV / "winning-args-corpus"
ROWS = CMV / "rows_deep.parquet"; PARENTS = CMV / "parents_deep.parquet"; VIEWS = CMV / "views_deep.npz"
RATERS = {"haiku": ("anthropic", "claude-haiku-4-5"), "openai": ("openai", "gpt-5-nano"), "jev": ("jev", "jev-latest")}
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "gpt-5-nano": (0.05, 0.40), "jev-latest": (0.042, 0.0)}   # USD per 1M input and output tokens (30 Sept 2026)
CONC = {"anthropic": 16, "openai": 8, "jev": 32}
CRITERIA = ["rude or hostile", "somewhat impolite or curt", "neutral", "polite", "very polite and courteous"]   # Jev rubric, levels 1 to 5
JEV_INSTR = "Rate the politeness of this online comment. Judge tone and courtesy only, not the opinion expressed."
def ratings_path(rater): return CMV / f"ratings_{rater}_deep.jsonl"
ALPHA, SEED, K, MAX_PER_PARENT, MIN_WORDS = 0.05, 0, 5, 8, 20
BOTS = {"DeltaBot", "AutoModerator", None, "[deleted]"}
STRAT, MARG = "stratified (parent identity)", "marginal (no conditioning)"
XS = {"politeness": "rating", "log_length": "log_len", "lexicon": "x_lex", "quote": "has_quote"}
YS = {"score": "y_score", "delta": "delta", "op_replied": "op_replied"}
PAIRS_MAIN = [("politeness", "score"), ("politeness", "delta"), ("log_length", "score"), ("log_length", "delta"), ("lexicon", "score"), ("lexicon", "delta")]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
def clean(text): return re.sub(r"\s+", " ", html.unescape(text or "")).strip()

# ------------------------------------------------------------------ build
def phase_build():
    convs = json.load(open(CORPUS / "conversations.json")); op_user = {cid: m.get("op-userID") for cid, m in convs.items()}
    ops = pd.DataFrame([{"op": cid, "text": clean((m.get("op-title") or "") + "\n\n" + (m.get("op-text-body") or ""))} for cid, m in convs.items()])
    U = {}; delta_bot = []
    with open(CORPUS / "utterances.jsonl") as f:
        for line in f:
            u = json.loads(line); m = u.get("meta") or {}; raw = u.get("text") or ""
            if u["id"] == u.get("root"): continue
            if u.get("user") == "DeltaBot":
                mm = re.search(r"delta awarded to /u/([\w-]+)", raw); delta_bot.append((u.get("reply-to"), u.get("root"), mm.group(1) if mm else None)); continue
            U[u["id"]] = {"id": u["id"], "root": u.get("root"), "parent": u.get("reply-to"), "user": u.get("user"), "ts": u.get("timestamp"), "raw": raw,
                          "score": m.get("score"), "deleted": int(raw.strip() in ("[deleted]", "[removed]") or u.get("user") in BOTS)}
    log(f"{len(U)} comments, {len(delta_bot)} DeltaBot confirmations, {len(ops)} posts")
    parent_of = {i: r["parent"] for i, r in U.items()}; user_of = {i: r["user"] for i, r in U.items()}
    awarded = set()                                   # the exact comment a delta was awarded to: the ancestor of the OP's delta comment written by the named user
    for rt, root, user in delta_bot:
        cur = parent_of.get(rt); steps = 0
        while cur is not None and cur != root and user_of.get(cur) != user and steps < 100: cur = parent_of.get(cur); steps += 1
        if cur is not None and cur != root and user_of.get(cur) == user: awarded.add(cur)
    n_children = {}; op_child = set()
    for i, r in U.items():
        if r["deleted"]: continue
        n_children[r["parent"]] = n_children.get(r["parent"], 0) + 1
        if r["user"] == op_user.get(r["root"]): op_child.add(r["parent"])
    def depth(i):
        d = 0; cur = i
        while cur is not None and cur in U and d < 100: cur = U[cur]["parent"]; d += 1
        return d                                      # 1 = top-level reply
    rows = []
    for i, r in U.items():
        p = U.get(r["parent"])
        if p is None or r["deleted"] or p["deleted"] or r["score"] is None: continue
        if r["user"] == op_user.get(r["root"]) or r["user"] == p["user"]: continue          # the original poster's replies and self-replies are not rows
        raw = r["raw"]; nw = len(raw.split()); pw = len(p["raw"].split())
        if nw < MIN_WORDS or pw < 5: continue
        rows.append({"id": i, "root": r["root"], "parent": r["parent"], "user": r["user"], "ts": r["ts"], "text": clean(raw), "n_words": nw,
                     "score": float(r["score"]), "delta": int(i in awarded), "op_replied": int(i in op_child), "has_quote": int(bool(re.search(r"(^|\n)\s*&gt;", raw))),
                     "has_url": int(bool(re.search(r"https?://|www\.", raw))), "n_children": n_children.get(i, 0)})
    df = pd.DataFrame(rows); df["ts"] = pd.to_numeric(df.ts, errors="coerce"); log(f"{len(df)} eligible deeper replies under {df.parent.nunique()} parents")
    df = df.sort_values(["parent", "ts"]).drop_duplicates(["parent", "user"], keep="first")
    df = df.sample(frac=1.0, random_state=SEED).groupby("parent", group_keys=False).head(MAX_PER_PARENT)
    df = df[df.groupby("parent").id.transform("size") >= 2].sort_values(["root", "parent", "ts"]).reset_index(drop=True)
    df["y_score"] = np.sign(df.score) * np.log1p(np.abs(df.score)); df["log_len"] = np.log(df.n_words)
    lex = np.array([lexicon(t) for t in df.text]); df["lex_polite"], df["lex_rude"], df["lex_q"], df["lex_excl"], df["lex_upper"] = lex.T; df["x_lex"] = df.lex_polite - df.lex_rude
    par = pd.DataFrame([{"parent": pid, "root": U[pid]["root"], "text": clean(U[pid]["raw"]), "user": U[pid]["user"], "by_op": int(U[pid]["user"] == op_user.get(U[pid]["root"])),
                         "depth": depth(pid), "score": float(U[pid]["score"]) if U[pid]["score"] is not None else np.nan, "n_words": len(U[pid]["raw"].split()),
                         "has_quote": int(bool(re.search(r"(^|\n)\s*&gt;", U[pid]["raw"]))), "has_url": int(bool(re.search(r"https?://|www\.", U[pid]["raw"]))),
                         "n_children": n_children.get(pid, 0)} for pid in df.parent.unique()])
    par["y_score"] = np.sign(par.score.fillna(0)) * np.log1p(np.abs(par.score.fillna(0))); par["log_words"] = np.log(par.n_words)
    ops = ops[ops.op.isin(df.root)].reset_index(drop=True)
    log(f"rows kept: {len(df)} under {df.parent.nunique()} parents in {df.root.nunique()} posts; delta rate {df.delta.mean():.4f}; OP replied {df.op_replied.mean():.3f}; "
        f"median score {df.score.median()}; parent words median {par.n_words.median()}, depth 1 share {(par.depth == 1).mean():.2f}")
    df.to_parquet(ROWS); par.to_parquet(PARENTS); ops.to_parquet(CMV / "ops_deep.parquet")
    json.dump({"n_rows": int(len(df)), "n_parents": int(df.parent.nunique()), "n_posts": int(df.root.nunique()), "delta_rate": float(df.delta.mean()), "n_delta": int(df.delta.sum()),
               "op_replied_rate": float(df.op_replied.mean()), "n_eligible": len(rows), "n_deltabot": len(delta_bot), "n_awarded_exact": len(awarded), "median_score": float(df.score.median()),
               "parent_words_median": float(par.n_words.median()), "parent_words_q90": float(par.n_words.quantile(.9)), "parent_depth1_share": float((par.depth == 1).mean()),
               "rows_per_parent_mean": float(len(df) / df.parent.nunique()), "max_per_parent": MAX_PER_PARENT, "min_words": MIN_WORDS}, open(RESULTS_DIR / "app_cmv_deep_build.json", "w"), indent=1)

# ------------------------------------------------------------------ embed
def phase_embed():
    par = pd.read_parquet(PARENTS); ops = pd.read_parquet(CMV / "ops_deep.parquet"); texts = list(par.text)
    views = {"E1": embed_len(texts, E1_NAME, 256, "parent"), "E2": embed_len(texts, E2_NAME, 384, "parent"), "S": surface(texts),
             "M": np.column_stack([par.log_words, par.y_score, par.by_op, par.depth, np.log1p(par.n_children), par.has_quote, par.has_url]).astype(float)}
    Tf = TfidfVectorizer(min_df=2, max_df=0.9, sublinear_tf=True, max_features=40000).fit_transform(texts); views["L300"] = TruncatedSVD(300, random_state=0).fit_transform(Tf)
    P1 = embed_len(list(ops.text), E1_NAME, 256, "op_deep"); pos = {o: i for i, o in enumerate(ops.op)}; views["P1"] = P1[[pos[r] for r in par.root]]
    np.savez(VIEWS, **views); log("views:", {k: v.shape for k, v in views.items()})
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(E1_NAME, cache_folder=str(CACHE_DIR / "st_models"), device="cpu")
    L = np.array([len(x) for x in m.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]])
    json.dump({"frac_over_256": float((L > 256).mean()), "frac_over_384": float((L > 384).mean()), "median_tokens": float(np.median(L))}, open(RESULTS_DIR / "app_cmv_deep_truncation.json", "w"))
    log(f"parent tokens: median {np.median(L):.0f}, frac>256 {(L>256).mean():.3f}, frac>384 {(L>384).mean():.3f}")

# ------------------------------------------------------------------ rate
def phase_rate(rater: str, budget_usd: float):
    """Politeness rating by one of three raters (same prompt as app_cmv.py for the two chat models; a five-level rubric for Jev).
    Resumable; a pilot of 20 calls extrapolates the cost against the budget. Keys from .env: ANTHROPIC_API_KEY, OPENAI_API_KEY, TYPESAFE_API_KEY."""
    import httpx
    provider, model = RATERS[rater]; path = ratings_path(rater); pin, pout = PRICES[model]
    key = os.environ.get({"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "jev": "TYPESAFE_API_KEY"}[provider])
    if not key: log(f"no API key for {rater}: skipping rating"); return
    df = pd.read_parquet(ROWS); done = set()
    if path.exists(): done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip() and json.loads(l).get("rating") is not None}
    todo = df[~df.id.isin(done)][["id", "text"]].values.tolist(); log(f"{rater} ({model}): {len(done)} cached, {len(todo)} to do")
    if not todo: return
    tot = {"in": 0, "out": 0}; cost = lambda: tot["in"] / 1e6 * pin + tot["out"] / 1e6 * pout
    async def call(client, rid, text):
        text = text[:6000]
        if provider == "anthropic":
            r = await client.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                                  json={"model": model, "max_tokens": 5, "temperature": 0.0, "system": SYSTEM, "messages": [{"role": "user", "content": PROMPT.format(text=text)}]})
            r.raise_for_status(); j = r.json(); out = "".join(b.get("text", "") for b in j["content"]); u = j["usage"]; tot["in"] += u["input_tokens"]; tot["out"] += u["output_tokens"]
            m = re.search(r"[1-5]", out); return {"id": rid, "rating": int(m.group()) if m else None, "raw": out[:10]}
        if provider == "openai":
            r = await client.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                  json={"model": model, "reasoning_effort": "minimal", "max_completion_tokens": 32,
                                        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT.format(text=text)}]})
            r.raise_for_status(); j = r.json(); out = j["choices"][0]["message"].get("content") or ""; u = j["usage"]; tot["in"] += u["prompt_tokens"]; tot["out"] += u["completion_tokens"]
            m = re.search(r"[1-5]", out); return {"id": rid, "rating": int(m.group()) if m else None, "raw": out[:10], "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        r = await client.post("https://api.typesafe.ai/v1/systemone", headers={"Authorization": f"Bearer {key}"},
                              json={"state": text, "model": model, "questions": {"politeness": {"type": "score", "instructions": JEV_INSTR, "criteria": CRITERIA}}})
        r.raise_for_status(); j = r.json(); a = j["answers"]["politeness"]; u = j.get("usage") or {}; tot["in"] += u.get("input_tokens", 0); tot["out"] += u.get("output_tokens", 0)
        return {"id": rid, "rating": float(a["score"]) + 1.0, "probabilities": a.get("probabilities"), "confidence": a.get("confidence"), "model": j.get("model")}
    async def one(client, sem, rid, text):
        async with sem:
            for attempt in range(5):
                try:
                    return await call(client, rid, text)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429, 500, 502, 503, 529) and attempt < 4:
                        ra = e.response.headers.get("retry-after"); await asyncio.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else 3.0 * (attempt + 1)); continue
                    return {"id": rid, "rating": None, "raw": f"ERR {e.response.status_code}: {e.response.text[:80]}"}
                except Exception as e:
                    if attempt == 4: return {"id": rid, "rating": None, "raw": f"ERR {type(e).__name__}: {str(e)[:60]}"}
                    await asyncio.sleep(2 * (attempt + 1))
    async def run(items, conc):
        async with httpx.AsyncClient(timeout=60) as client:
            sem = asyncio.Semaphore(conc)
            with open(path, "a") as f:
                for k in range(0, len(items), 400):
                    res = await asyncio.gather(*[one(client, sem, rid, text) for rid, text in items[k:k + 400]])
                    for r in res: f.write(json.dumps(r) + "\n")
                    nf = sum(r["rating"] is None for r in res); log(f"  rated {min(k+400, len(items))}/{len(items)}, failures {nf}, running cost {cost():.3f} USD" + (f"  e.g. {next(r['raw'] for r in res if r['rating'] is None)}" if nf else ""))
    pilot = todo[:20]; asyncio.run(run(pilot, 8)); per_call = cost() / max(len(pilot), 1); est = per_call * len(todo)
    log(f"pilot: {per_call*1000:.4f} USD per 1000 calls; extrapolated {est:.2f} USD for {len(todo)} calls")
    if est > budget_usd: log(f"ABORT rating: estimate {est:.2f} USD exceeds budget {budget_usd} USD"); return
    asyncio.run(run(todo[20:], CONC[provider]))
    usage_path = RESULTS_DIR / f"app_cmv_deep_rating_usage_{rater}.json"; prev = json.load(open(usage_path)) if usage_path.exists() else {}
    n_ok = sum(1 for l in path.read_text().splitlines() if l.strip() and json.loads(l).get("rating") is not None)
    json.dump({"rater": rater, "model": model, "n_calls": prev.get("n_calls", 0) + len(todo), "n_rated": n_ok, "tokens": {k: prev.get("tokens", {}).get(k, 0) + v for k, v in tot.items()},
               "cost_usd": prev.get("cost_usd", 0.0) + cost(), "cost_usd_this_run": cost(), "prices_per_M": [pin, pout]}, open(usage_path, "w"))
    log(f"rating done: {cost():.3f} USD")

def read_ratings(rater):
    path = ratings_path(rater)
    if not path.exists(): return pd.DataFrame(columns=["id", "rating"])
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()]).dropna(subset=["rating"]).drop_duplicates("id")

def phase_agree(rater: str):
    """Agreement of the rater with the Haiku ratings on the rows both rated (Haiku rated 4,728 of the deeper replies on 30 Sept 2026)."""
    from scipy.stats import spearmanr
    a = read_ratings("haiku")[["id", "rating"]].rename(columns={"rating": "haiku"}); b = read_ratings(rater)[["id", "rating"]].rename(columns={"rating": "r"}); m = a.merge(b, on="id")
    if len(m) < 30: log(f"agreement: only {len(m)} rows rated by both"); return
    out = {"rater": rater, "n_overlap": int(len(m)), "pearson": float(np.corrcoef(m.haiku, m.r)[0, 1]), "spearman": float(spearmanr(m.haiku, m.r).correlation),
           "mean_haiku": float(m.haiku.mean()), "mean_rater": float(m.r.mean()), "sd_haiku": float(m.haiku.std()), "sd_rater": float(m.r.std()),
           "exact_agreement": float((np.round(m.r) == m.haiku).mean()), "within_one": float((np.abs(np.round(m.r) - m.haiku) <= 1).mean()),
           "dist_haiku": m.haiku.value_counts(normalize=True).sort_index().round(3).to_dict(), "dist_rater_rounded": pd.Series(np.round(m.r)).value_counts(normalize=True).sort_index().round(3).to_dict()}
    json.dump(out, open(RESULTS_DIR / f"app_cmv_deep_rater_agreement_{rater}.json", "w"), indent=1, default=float); log("agreement:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if not k.startswith("dist")})

# ------------------------------------------------------------------ test
def within(x, groups): return x - pd.Series(x).groupby(groups).transform("mean").values
def stratified(X, Y, strata, clusters, B=1000, seed=SEED):
    """Within-stratum demeaned correlation with clustered se, and an exact permutation p-value (X permuted within strata)."""
    xt, yt = within(X, strata), within(Y, strata); st = corr_stats(xt, yt, clusters)
    rng = np.random.default_rng(seed); g = pd.factorize(strata)[0]; order = np.argsort(g, kind="stable"); gs = g[order].astype(float); xs, ys = xt[order], yt[order]
    denom = np.sqrt(np.sum(xt**2) * np.sum(yt**2)); rho_b = np.empty(B)
    for b in range(B):
        perm = np.argsort(gs + rng.random(len(gs)), kind="stable"); rho_b[b] = np.sum(xs[perm] * ys) / denom   # a random order within each stratum
    st["perm_p"] = float((1 + np.sum(np.abs(rho_b) >= abs(st["rho"]))) / (B + 1)); st["perm_sd"] = float(rho_b.std(ddof=1)); return st

def load_rows(rater):
    df = pd.read_parquet(ROWS); rat = read_ratings(rater)
    return df.merge(rat[["id", "rating"]], on="id", how="left") if len(rat) else df.assign(rating=np.nan)

def prog_model():
    from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))   # standardized inputs: the singular directions are small in scale

def phase_test(rater, regressor="ols", smoke=False, views=None, bench=True):
    if regressor == "mlp": return phase_test_nn(rater, smoke, views, bench)
    df = load_rows(rater); par = pd.read_parquet(PARENTS); V = dict(np.load(VIEWS)); log(f"rows {len(df)}, ratings {df.rating.notna().sum()}")
    pidx = {p: i for i, p in enumerate(par.parent)}; idx = np.array([pidx[p] for p in df.parent]); strata = df.parent.values; clusters = df.root.values
    Dense = np.hstack([V["L300"], V["S"], V["M"]])[idx]
    results, bench = [], []
    for xn, xcol in XS.items():
        X = df[xcol].values.astype(float); ok = ~np.isnan(X)
        if ok.sum() < 100: log(f"skip {xn}: too few values"); continue
        for yn, ycol in YS.items():
            Y = df[ycol].values.astype(float); m = ok & ~np.isnan(Y); Xm, Ym, sm, cm, im = X[m], Y[m], strata[m], clusters[m], idx[m]
            keep = pd.Series(sm).groupby(sm).transform("size").values >= 2; Xm, Ym, sm, cm, im = Xm[keep], Ym[keep], sm[keep], cm[keep], im[keep]
            fm = list(GroupKFold(K).split(np.zeros(len(Xm)), groups=cm)); n = len(Xm)
            st = stratified(Xm, Ym, sm, cm); results.append({"X": xn, "Y": yn, "psi": STRAT, "n": n, **st}); log(f"{xn}/{yn} stratified: rho={st['rho']:.4f} se={st['se']:.4f} perm_p={st['perm_p']:.3f}")
            mg = corr_stats(Xm - Xm.mean(), Ym - Ym.mean(), cm); results.append({"X": xn, "Y": yn, "psi": MARG, "n": n, **mg}); log(f"{xn}/{yn} marginal: rho={mg['rho']:.4f} se={mg['se']:.4f}")
            Dm = Dense[im]; predX = np.empty(n); predY = np.empty(n)
            for tr, te in fm:
                predX[te] = prog_model().fit(Dm[tr], Xm[tr]).predict(Dm[te]); predY[te] = prog_model().fit(Dm[tr], Ym[tr]).predict(Dm[te])
            prog = np.c_[predX, predY]
            psis = {"E1": V["E1"][im], "E2": V["E2"][im], "S": V["S"][im], "M": V["M"][im], "L300": V["L300"][im], "P1": V["P1"][im],
                    "[E1,M]": np.hstack([V["E1"][im], V["M"][im]]), "[E1,P1]": np.hstack([V["E1"][im], V["P1"][im]]), "[E1,L300]": np.hstack([V["E1"][im], V["L300"][im]]),
                    "[E1,E2]": np.hstack([V["E1"][im], V["E2"][im]]), "prognostic": prog, "[E1,prognostic]": np.hstack([V["E1"][im], prog])}
            for pn, Psi in psis.items():
                if not config_feasible(n, Psi.shape[1], K): log(f"  {pn}: not reported ({n} rows, {Psi.shape[1]} columns)"); continue
                rA = Xm - fit_predict_crossfit(Psi, Xm, K, SEED, regressor, fm); rB = Ym - fit_predict_crossfit(Psi, Ym, K, SEED, regressor, fm)
                r = corr_stats(rA, rB, cm); r.update({"R2_X": float(1 - np.mean(rA**2) / np.var(Xm)), "R2_Y": float(1 - np.mean(rB**2) / np.var(Ym))})
                results.append({"X": xn, "Y": yn, "psi": pn, "n": n, **r}); log(f"{xn}/{yn} psi={pn}: rho={r['rho']:.4f} se={r['se']:.4f} RVa={r['RV_alpha']:.4f} R2X={r['R2_X']:.3f} R2Y={r['R2_Y']:.3f}")
            for pn in ("E2", "S", "M", "L300", "P1", "prognostic"):
                Pp = prog if pn == "prognostic" else V[pn][im]
                bA = partial_r2_benchmark(Xm, V["E1"][im], Pp, regressor, K, SEED, fm); bB = partial_r2_benchmark(Ym, V["E1"][im], Pp, regressor, K, SEED, fm)
                bench.append({"X": xn, "Y": yn, "view": pn, "r2_X": bA["r2"], "r2_Y": bB["r2"], "rho_bench": float(np.sqrt(bA["r2"] * bB["r2"]))}); log(f"  bench E1->{pn}: r2X={bA['r2']:.4f} r2Y={bB['r2']:.4f} rho_bench={np.sqrt(bA['r2']*bB['r2']):.4f}")
            pd.DataFrame(results).to_csv(RESULTS_DIR / f"app_cmv_deep_results_{rater}.csv", index=False); pd.DataFrame(bench).to_csv(RESULTS_DIR / f"app_cmv_deep_bench_{rater}.csv", index=False)
    log("tests done")

# ------------------------------------------------------------------ test with the networks (the paper's application, 7 Oct 2026)
PARTNERS = ("E2", "S", "M", "L300", "P1", "prognostic"); COMBO = {"E2": "[E1,E2]", "M": "[E1,M]", "P1": "[E1,P1]", "L300": "[E1,L300]", "prognostic": "[E1,prognostic]"}
PAIRS_ALL_VIEWS = [("politeness", "score"), ("log_length", "score")]
def phase_test_nn(rater, smoke=False, views=None, bench=True):
    """The tests of phase_test with the small networks of citext.nn as the regressor: the six main pairs, every embedding for the two score
    pairs and E1 for the other four, and the six benchmarks of E1 for every pair. X and Y are fitted in one call, so their networks have
    different seeds; a benchmark reuses the fit of the combined embedding where the views include it. Resumable per pair: a pair is complete
    when its six benchmarks are in app_cmv_deep_bench_<rater>_nn.csv, and the rows of an unfinished pair are redone."""
    kw = {"configs": "small", "seeds": 1} if smoke else {}; sfx = "_nn_smoke" if smoke else "_nn"
    fo, fb = RESULTS_DIR / f"app_cmv_deep_results_{rater}{sfx}.csv", RESULTS_DIR / f"app_cmv_deep_bench_{rater}{sfx}.csv"
    df = load_rows(rater); par = pd.read_parquet(PARENTS); V = dict(np.load(VIEWS)); log(f"rows {len(df)}, ratings {df.rating.notna().sum()}; regressor: networks")
    pidx = {p: i for i, p in enumerate(par.parent)}; idx = np.array([pidx[p] for p in df.parent]); strata = df.parent.values; clusters = df.root.values
    Dense = np.hstack([V["L300"], V["S"], V["M"]])[idx]; partners = (PARTNERS[:1] if smoke else PARTNERS) if bench else ()
    results = pd.read_csv(fo).to_dict("records") if fo.exists() and not smoke else []; bench = pd.read_csv(fb).to_dict("records") if fb.exists() and not smoke else []
    have = {(r["X"], r["Y"], r["psi"]) for r in results}
    done = {pr for pr in PAIRS_MAIN if (sum((b["X"], b["Y"]) == pr for b in bench) == len(partners) if partners else all((*pr, v) in have for v in (views or ["E1"])))}
    bench = [b for b in bench if (b["X"], b["Y"]) in done]   # rows of an unfinished pair are kept and their fits skipped; its benchmarks are redone from the stored R^2 values
    for xn, yn in (PAIRS_MAIN[:1] if smoke else PAIRS_MAIN):
        if (xn, yn) in done: log(f"{xn}/{yn}: complete in {fo.name}, skipped"); continue
        X = df[XS[xn]].values.astype(float); Y = df[YS[yn]].values.astype(float); m = ~np.isnan(X) & ~np.isnan(Y); Xm, Ym, sm, cm, im = X[m], Y[m], strata[m], clusters[m], idx[m]
        keep = pd.Series(sm).groupby(sm).transform("size").values >= 2; Xm, Ym, sm, cm, im = Xm[keep], Ym[keep], sm[keep], cm[keep], im[keep]
        fm = list(GroupKFold(K).split(np.zeros(len(Xm)), groups=cm)); n = len(Xm)
        old = {r["psi"]: r for r in results if (r["X"], r["Y"]) == (xn, yn)}
        if STRAT not in old: st = stratified(Xm, Ym, sm, cm); results.append({"X": xn, "Y": yn, "psi": STRAT, "n": n, **st}); log(f"{xn}/{yn} stratified: rho={st['rho']:.4f} se={st['se']:.4f} perm_p={st['perm_p']:.3f}")
        if MARG not in old: mg = corr_stats(Xm - Xm.mean(), Ym - Ym.mean(), cm); results.append({"X": xn, "Y": yn, "psi": MARG, "n": n, **mg}); log(f"{xn}/{yn} marginal: rho={mg['rho']:.4f} se={mg['se']:.4f}")
        Dm = Dense[im]; predX = np.empty(n); predY = np.empty(n)
        for tr, te in fm:
            predX[te] = prog_model().fit(Dm[tr], Xm[tr]).predict(Dm[te]); predY[te] = prog_model().fit(Dm[tr], Ym[tr]).predict(Dm[te])
        prog = np.c_[predX, predY]
        psis = {"E1": V["E1"][im], "E2": V["E2"][im], "S": V["S"][im], "M": V["M"][im], "L300": V["L300"][im], "P1": V["P1"][im],
                "[E1,M]": np.hstack([V["E1"][im], V["M"][im]]), "[E1,P1]": np.hstack([V["E1"][im], V["P1"][im]]), "[E1,L300]": np.hstack([V["E1"][im], V["L300"][im]]),
                "[E1,E2]": np.hstack([V["E1"][im], V["E2"][im]]), "prognostic": prog, "[E1,prognostic]": np.hstack([V["E1"][im], prog])}
        names = list(views) if views else (list(psis) if (xn, yn) in PAIRS_ALL_VIEWS and not smoke else ["E1"]); vX, vY = float(np.var(Xm)), float(np.var(Ym))
        mse = {pn: ((1 - old[pn]["R2_X"]) * vX, (1 - old[pn]["R2_Y"]) * vY) for pn in names if pn in old}   # residual mean squares of the fits already in the file
        for pn in names:
            if pn in mse: log(f"{xn}/{yn} psi={pn}: in {fo.name}, skipped"); continue
            Psi = psis[pn]; Mh, nfo = crossfit_nn(Psi, np.c_[Xm, Ym], fm, **kw); rA, rB = Xm - Mh[:, 0], Ym - Mh[:, 1]; mse[pn] = (float(np.mean(rA**2)), float(np.mean(rB**2)))
            r = corr_stats(rA, rB, cm); r.update({"R2_X": float(1 - np.mean(rA**2) / np.var(Xm)), "R2_Y": float(1 - np.mean(rB**2) / np.var(Ym)), "configs_X": cfg_summary(nfo, [0]), "configs_Y": cfg_summary(nfo, [1]), "val_r2": nfo["val_r2"], "iters": nfo["iters"]})
            results.append({"X": xn, "Y": yn, "psi": pn, "n": n, **r}); pd.DataFrame(results).to_csv(fo, index=False)
            log(f"{xn}/{yn} psi={pn} (d={Psi.shape[1]}): rho={r['rho']:.4f} se={r['se']:.4f} RVa={r['RV_alpha']:.4f} R2X={r['R2_X']:.3f} R2Y={r['R2_Y']:.3f} | configs {r['configs_X']} / {r['configs_Y']}, val R2 {r['val_r2']:.3f}, iters {r['iters']:.0f}")
        for pn in partners:
            mA1, mB1 = mse["E1"]
            if pn in COMBO and COMBO[pn] in mse: mA2, mB2 = mse[COMBO[pn]]
            else:
                Pp = prog if pn == "prognostic" else V[pn][im]; Mh, _ = crossfit_nn(np.hstack([V["E1"][im], Pp]), np.c_[Xm, Ym], fm, **kw); mA2, mB2 = float(np.mean((Xm - Mh[:, 0])**2)), float(np.mean((Ym - Mh[:, 1])**2))
            bA = max(0.0, 1 - mA2 / mA1); bB = max(0.0, 1 - mB2 / mB1)
            bench.append({"X": xn, "Y": yn, "view": pn, "r2_X": bA, "r2_Y": bB, "rho_bench": float(np.sqrt(bA * bB))}); pd.DataFrame(bench).to_csv(fb, index=False)
            log(f"  bench E1->{pn}: r2X={bA:.4f} r2Y={bB:.4f} rho_bench={np.sqrt(bA*bB):.4f}")
    log("tests done (networks)")

# ------------------------------------------------------------------ components (as app_cmv_components.py, strata = parents)
def anova(df, cols, g):
    N, G = len(df), df[g].nunique(); ng = df.groupby(g).size().values; n0 = (N - np.sum(ng**2) / N) / (G - 1)
    means = df.groupby(g)[cols].transform("mean"); grand = df[cols].mean(); dev_w = df[cols] - means; dev_b = means - grand; out = {}
    for a in cols:
        for b in cols:
            if cols.index(b) < cols.index(a): continue
            ssb = float((dev_b[a] * dev_b[b]).sum()); ssw = float((dev_w[a] * dev_w[b]).sum()); msb, msw = ssb / (G - 1), ssw / (N - G)
            out[(a, b)] = {"between": (msb - msw) / n0, "within": msw}
    return out, {"N": int(N), "G": int(G), "n0": float(n0)}

def phase_components(rater, sfx=""):
    df = load_rows(rater); res = pd.read_csv(RESULTS_DIR / f"app_cmv_deep_results_{rater}{sfx}.csv"); out = {}
    for xn, xcol in XS.items():
        for yn, ycol in YS.items():
            d = df.dropna(subset=[xcol, ycol]).copy(); d = d[d.groupby("parent")["id"].transform("size") >= 2]
            if len(d) < 100: continue
            comp, meta = anova(d, [xcol, ycol] if xcol != ycol else [xcol], "parent"); vx, vy, cxy = comp[(xcol, xcol)], comp[(ycol, ycol)], comp[(xcol, ycol)]
            share_x = vx["between"] / (vx["between"] + vx["within"]); share_y = vy["between"] / (vy["between"] + vy["within"])
            lam = cxy["between"] / np.sqrt(vx["between"] * vy["between"]) if min(vx["between"], vy["between"]) > 0 else float("nan")
            post, _ = anova(d, [xcol, ycol], "root"); px, py = post[(xcol, xcol)], post[(ycol, ycol)]
            rec = {**meta, "share_between_X": share_x, "share_between_Y": share_y, "lambda_hat": float(lam), "within_corr": cxy["within"] / np.sqrt(vx["within"] * vy["within"]),
                   "share_post_X": px["between"] / (px["between"] + px["within"]), "share_post_Y": py["between"] / (py["between"] + py["within"])}
            r = res[(res.X == xn) & (res.Y == yn)].set_index("psi")
            for psi in ("E1", "E2", "[E1,M]", "[E1,E2]", "prognostic", "[E1,prognostic]"):
                if psi in r.index and STRAT in r.index:
                    r2x, r2y = max(0.0, float(r.loc[psi, "R2_X"])), max(0.0, float(r.loc[psi, "R2_Y"])); R2A, R2B = max(share_x - r2x, 0.0), max(share_y - r2y, 0.0)
                    R2A_def, R2B_def = R2A / max(1.0 - r2x, 1e-12), R2B / max(1.0 - r2y, 1e-12); rho_s, rho_e = float(r.loc[STRAT, "rho"]), float(r.loc[psi, "rho"])
                    pred = rho_s * np.sqrt((1 - R2A_def) * (1 - R2B_def)) + lam * np.sqrt(R2A_def * R2B_def)
                    rec[psi] = {"emb_R2_X": r2x, "emb_R2_Y": r2y, "R2_A_def": float(R2A_def), "R2_B_def": float(R2B_def), "bound_def": float(np.sqrt(R2A_def * R2B_def)),
                                "predicted_rho": float(pred), "predicted_bias": float(pred - rho_s), "measured_bias": rho_e - rho_s, "rho_strat": rho_s, "rho_psi": rho_e,
                                "se_psi": float(r.loc[psi, "se"]), "se_strat": float(r.loc[STRAT, "se"]), "RV_alpha": float(r.loc[psi, "RV_alpha"])}
            out[f"{xn}/{yn}"] = rec
            e1 = rec.get("E1", {}); log(f"{xn}/{yn}: N={meta['N']} G={meta['G']} shareX={share_x:.3f} shareY={share_y:.3f} (post: {rec['share_post_X']:.3f}/{rec['share_post_Y']:.3f}) lambda={lam:.3f}"
                                     + (f" E1: pred_bias={e1['predicted_bias']:.4f} meas_bias={e1['measured_bias']:.4f} rho_id={e1['rho_strat']:.4f} rho_E1={e1['rho_psi']:.4f}" if e1 else ""))
    json.dump(out, open(RESULTS_DIR / f"app_cmv_deep_components_{rater}{sfx}.json", "w"), indent=1)

# ------------------------------------------------------------------ plots
def phase_plots(rater, sfx="", fsfx=""):
    res = pd.read_csv(RESULTS_DIR / f"app_cmv_deep_results_{rater}{sfx}.csv"); fb = RESULTS_DIR / f"app_cmv_deep_bench_{rater}{sfx}.csv"; ben = pd.read_csv(fb) if fb.exists() else None
    pairs = [p for p in PAIRS_MAIN if ((res.X == p[0]) & (res.Y == p[1])).any()]
    xn = {"politeness": "politeness", "log_length": "log length", "lexicon": "lexicon", "quote": "quote"}
    series = [(MARG, r"no conditioning, $\hat\rho_0$", P.MUTED, "D", -0.22), (STRAT, r"stratified reference $\hat\rho_{\mathrm{id}}$", P.CAT[2], "s", 0.0), ("E1", r"embedded test, $\psi = E_1$", P.CAT[0], "o", 0.22)]
    P.setup(); fig, ax = plt.subplots(figsize=(P.SINGLE[0], 2.0)); xs = np.arange(len(pairs))
    for psi, name, col, mk, dx in series:
        rows = [res[(res.X == x) & (res.Y == y) & (res.psi == psi)].iloc[0] for x, y in pairs]
        ax.errorbar(xs + dx, [r.rho for r in rows], yerr=1.96 * np.array([r.se for r in rows]), fmt=mk, ms=3.8, color=col, capsize=2, lw=1, label=name)
    ax.axhline(0, color=P.AXIS, lw=0.6); ax.set_xticks(xs); ax.set_xticklabels([f"{xn[x]}\n{y}" for x, y in pairs], fontsize=6.3); ax.set_ylabel(r"$\hat\rho$ with 95% interval")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, frameon=False, handletextpad=0.3, columnspacing=0.8); P.savefig(fig, f"app_cmv_deep_main_{rater}{fsfx}.pdf"); log(f"app_cmv_deep_main_{rater}{fsfx}.pdf")
    if ben is None: log(f"no benchmark file {fb.name}: the embeddings figure is skipped"); return
    order = ["E1", "E2", "[E1,E2]", "S", "M", "L300", "P1", "[E1,M]", "[E1,P1]", "[E1,L300]", "prognostic", "[E1,prognostic]"]
    labels = {"E1": "$E_1$", "E2": "$E_2$", "[E1,E2]": "$[E_1,E_2]$", "S": "$S$", "M": "$M$", "L300": "$L$", "P1": "$P$", "[E1,M]": "$[E_1,M]$", "[E1,P1]": "$[E_1,P]$", "[E1,L300]": "$[E_1,L]$", "prognostic": "prog.", "[E1,prognostic]": "$[E_1,$prog.$]$"}
    view_of = {"E2": "E2", "[E1,E2]": "E2", "S": "S", "M": "M", "L300": "L300", "P1": "P1", "[E1,M]": "M", "[E1,P1]": "P1", "[E1,L300]": "L300", "prognostic": "prognostic", "[E1,prognostic]": "prognostic"}
    vp = [p for p in [("politeness", "score"), ("log_length", "score")] if ((res.X == p[0]) & (res.Y == p[1])).any()]
    fig, axs = plt.subplots(1, len(vp), figsize=(6.6, 2.5))
    for ax, (x, y) in zip(np.atleast_1d(axs), vp):
        r = res[(res.X == x) & (res.Y == y)].set_index("psi"); b = ben[(ben.X == x) & (ben.Y == y)].set_index("view"); st = r.loc[STRAT]; ks = [k for k in order if k in r.index]
        ax.axhspan(st.rho - 1.96 * st.se, st.rho + 1.96 * st.se, color=P.CAT[2], alpha=0.18, lw=0); ax.axhline(st.rho, color=P.CAT[2], lw=1.2, label=r"stratified reference $\hat\rho_{\mathrm{id}}$")
        ax.axhline(r.loc[MARG, "rho"], color=P.MUTED, lw=0.9, ls=":", label=r"no conditioning, $\hat\rho_0$")
        xs = np.arange(len(ks)); ax.errorbar(xs, [r.loc[k, "rho"] for k in ks], yerr=1.96 * np.array([r.loc[k, "se"] for k in ks]), fmt="o", color=P.CAT[0], capsize=2, label=r"embedded $\hat\rho_\psi$")
        bx = [i for i, k in enumerate(ks) if k in view_of]; ax.scatter(bx, [b.loc[view_of[ks[i]], "rho_bench"] for i in bx], marker="s", color=P.CAT[1], s=22, zorder=3, label=r"$\hat\rho_{E_1 \to \psi'}$")
        ax.axhline(0, color=P.AXIS, lw=0.6); ax.set_xticks(xs); ax.set_xticklabels([labels[k] for k in ks], fontsize=6, rotation=30, ha="right"); ax.set_title(f"$X$ = {xn[x]}, $Y$ = {y}"); ax.set_ylabel(r"$\hat\rho$")
    np.atleast_1d(axs)[0].legend(loc="best", fontsize=6); P.savefig(fig, f"app_cmv_deep_views_{rater}{fsfx}.pdf"); log(f"app_cmv_deep_views_{rater}{fsfx}.pdf")

# ------------------------------------------------------------------ macros
def phase_macros(rater: str, write: bool, sfx="", regressor="ols"):
    res = pd.read_csv(RESULTS_DIR / f"app_cmv_deep_results_{rater}{sfx}.csv"); ben = pd.read_csv(RESULTS_DIR / f"app_cmv_deep_bench_{rater}{sfx}.csv"); comp = json.load(open(RESULTS_DIR / f"app_cmv_deep_components_{rater}{sfx}.json"))
    build = json.load(open(RESULTS_DIR / "app_cmv_deep_build.json")); up = RESULTS_DIR / f"app_cmv_deep_rating_usage_{rater}.json"; usage = json.load(open(up)) if up.exists() else {}
    agp = RESULTS_DIR / f"app_cmv_deep_rater_agreement_{rater}.json"; ag = json.load(open(agp)) if agp.exists() else {}
    def sub(x, y): return res[(res.X == x) & (res.Y == y)].set_index("psi")
    def emb(x, y): r = sub(x, y); return r[~r.index.isin([STRAT, MARG])]
    f3 = lambda v: f"{v:.3f}"; pct = lambda v, nd=0: f"{100*v:.{nd}f}\\%"; th = lambda n: f"{n:,}".replace(",", "{,}")
    ps = sub("politeness", "score") if "politeness" in res.X.values else sub("log_length", "score"); st = ps.loc[STRAT]; c = comp.get("politeness/score", comp["log_length/score"]); e1 = c["E1"]
    ls = sub("log_length", "score"); cl = comp["log_length/score"]; cle = cl["E1"]
    M = {"resDeepN": th(build["n_rows"]), "resDeepNparents": th(build["n_parents"]), "resDeepNposts": th(build["n_posts"]), "resDeepMaxPerParent": str(build["max_per_parent"]),
         "resDeepRowsPerParent": f"{build['rows_per_parent_mean']:.1f}", "resDeepDeltaRate": pct(build["delta_rate"], 1), "resDeepOpRepliedRate": pct(build["op_replied_rate"]),
         "resDeepParentWords": f"{build['parent_words_median']:.0f}", "resDeepParentDepthOne": pct(build["parent_depth1_share"]),
         "resDeepRatingCost": f"{usage['cost_usd']:.1f}" if usage else "TBD", "resDeepRated": th(usage["n_rated"]) if usage else "TBD",
         "resDeepPolStrat": f3(st.rho), "resDeepPolStratSe": f3(st.se), "resDeepPolMarg": f3(ps.loc[MARG, "rho"]), "resDeepPolEone": f3(ps.loc["E1", "rho"]), "resDeepPolEoneSe": f3(ps.loc["E1", "se"]),
         "resDeepPolEmbMin": f3(emb(*(("politeness", "score") if "politeness" in res.X.values else ("log_length", "score"))).rho.min()), "resDeepPolEmbMax": f3(emb(*(("politeness", "score") if "politeness" in res.X.values else ("log_length", "score"))).rho.max()),
         "resDeepPolShareX": pct(c["share_between_X"]), "resDeepPolShareY": pct(c["share_between_Y"]), "resDeepPolSharePostX": pct(c["share_post_X"]), "resDeepPolSharePostY": pct(c["share_post_Y"]),
         "resDeepPolLambda": f"{c['lambda_hat']:.2f}", "resDeepPolEmbRtwoX": pct(e1["emb_R2_X"], 1), "resDeepPolEmbRtwoY": pct(e1["emb_R2_Y"], 1),
         "resDeepPolRtwoA": f3(e1["R2_A_def"]), "resDeepPolRtwoB": f3(e1["R2_B_def"]), "resDeepPolBound": f3(e1["bound_def"]), "resDeepPolPredRho": f3(e1["predicted_rho"]), "resDeepPolPredBias": f3(e1["predicted_bias"]),
         "resDeepPolMeasBias": f3(e1["measured_bias"]), "resDeepPolRValpha": f"{e1['RV_alpha']:.3f}",
         "resDeepLenStrat": f3(ls.loc[STRAT, "rho"]), "resDeepLenStratSe": f3(ls.loc[STRAT, "se"]), "resDeepLenMarg": f3(ls.loc[MARG, "rho"]), "resDeepLenEone": f3(ls.loc["E1", "rho"]), "resDeepLenEoneSe": f3(ls.loc["E1", "se"]),
         "resDeepLenShareX": pct(cl["share_between_X"]), "resDeepLenShareY": pct(cl["share_between_Y"]), "resDeepLenLambda": f"{cl['lambda_hat']:.2f}", "resDeepLenEmbRtwoX": pct(cle["emb_R2_X"], 1), "resDeepLenEmbRtwoY": pct(cle["emb_R2_Y"], 1),
         "resDeepLenBound": f3(cle["bound_def"]), "resDeepLenPredRho": f3(cle["predicted_rho"]), "resDeepLenPredBias": f3(cle["predicted_bias"]), "resDeepLenMeasBias": f3(cle["measured_bias"]), "resDeepLenRValpha": f"{cle['RV_alpha']:.3f}",
         "resDeepBenchMax": f3(ben[ben.X.isin(["politeness", "log_length", "lexicon"]) & ben.Y.isin(["score", "delta"])].rho_bench.max()), "resDeepBenchMaxAll": f3(ben.rho_bench.max()),
         "resDeepPredMeasMaxDiff": f3(max(abs(comp[f"{x}/{y}"]["E1"]["predicted_bias"] - comp[f"{x}/{y}"]["E1"]["measured_bias"]) for x, y in PAIRS_MAIN if f"{x}/{y}" in comp and "E1" in comp[f"{x}/{y}"]) + 0.0005), "resDeepRater": {"openai": "GPT-5 nano", "jev": "Jev", "haiku": "Claude Haiku 4.5"}[rater],
         "resDeepParentFit": pct(1 - json.load(open(RESULTS_DIR / "app_cmv_deep_truncation.json"))["frac_over_256"]),
         "resDeepAgreeN": th(ag["n_overlap"]) if ag else "TBD", "resDeepAgreeCorr": f"{ag['pearson']:.2f}" if ag else "TBD", "resDeepAgreeSpearman": f"{ag['spearman']:.2f}" if ag else "TBD",
         "resDeepAgreeExact": pct(ag["exact_agreement"]) if ag else "TBD", "resDeepAgreeWithinOne": pct(ag["within_one"]) if ag else "TBD",
         "resDeepAgreeMeanRater": f"{ag['mean_rater']:.2f}" if ag else "TBD", "resDeepAgreeMeanHaiku": f"{ag['mean_haiku']:.2f}" if ag else "TBD",
         "resDeepBiasMax": f3(max(abs(emb(x, y).rho - sub(x, y).loc[STRAT, "rho"]).max() for x, y in PAIRS_MAIN if ((res.X == x) & (res.Y == y)).any()))}
    for x, y in PAIRS_MAIN:
        if ((res.X == x) & (res.Y == y)).any():
            k = {"politeness": "Pol", "log_length": "Len", "lexicon": "Lex"}[x] + {"score": "Score", "delta": "Delta"}[y]; r = sub(x, y)
            M[f"resDeep{k}Gap"] = f3(r.loc[MARG, "rho"] - r.loc[STRAT, "rho"]); M[f"resDeep{k}EoneBias"] = f3(r.loc["E1", "rho"] - r.loc[STRAT, "rho"])
    rr = read_ratings(rater); hh = read_ratings("haiku")
    if len(rr):
        M["resDeepRaterMedian"] = f"{rr.rating.median():.2f}"; M["resDeepRaterSd"] = f"{rr.rating.std():.2f}"
        ov = hh[["id", "rating"]].rename(columns={"rating": "haiku"}).merge(rr[["id", "rating"]], on="id")
        if len(ov): g = ov.groupby("haiku").rating.mean(); M["resDeepRaterAtHaikuOne"] = f"{g.get(1, float('nan')):.2f}"; M["resDeepRaterAtHaikuFive"] = f"{g.get(5, float('nan')):.2f}"
    M["resDeepRegressor"] = {"ols": "least squares", "mlp": "small neural networks"}[regressor]
    fl = RESULTS_DIR / f"app_cmv_deep_results_{rater}.csv"
    if regressor == "mlp" and fl.exists():   # least squares on the same folds (the comparison, suffix Ls), restricted to the pairs and embeddings of the network run
        rl = pd.read_csv(fl); bl = pd.read_csv(RESULTS_DIR / f"app_cmv_deep_bench_{rater}.csv"); strat = {(x, y): float(rl[(rl.X == x) & (rl.Y == y) & (rl.psi == STRAT)].rho.iloc[0]) for x, y in PAIRS_MAIN if ((rl.X == x) & (rl.Y == y)).any()}
        both = res[~res.psi.isin([STRAT, MARG])].merge(rl, on=["X", "Y", "psi"], suffixes=("", "_ls"))
        M["resDeepRhoDiffMaxLs"] = f3((both.rho - both.rho_ls).abs().max()); M["resDeepRhoDiffMaxLsN"] = str(len(both)); M["resDeepBiasMaxLs"] = f3(max(abs(r.rho_ls - strat[(r.X, r.Y)]) for r in both.itertuples()))
        bb = ben.merge(bl, on=["X", "Y", "view"], suffixes=("", "_ls")); M["resDeepBenchMaxLs"] = f3(bb.rho_bench_ls.max()); M["resDeepBenchDiffMaxLs"] = f3((bb.rho_bench - bb.rho_bench_ls).abs().max())
        for (x, y), k in ((("politeness", "score"), "Pol"), (("log_length", "score"), "Len")):
            r = rl[(rl.X == x) & (rl.Y == y)].set_index("psi")
            if "E1" in r.index: M[f"resDeep{k}EoneLs"] = f3(r.loc["E1", "rho"]); M[f"resDeep{k}EmbRtwoXLs"] = pct(max(r.loc["E1", "R2_X"], 0), 1); M[f"resDeep{k}EmbRtwoYLs"] = pct(max(r.loc["E1", "R2_Y"], 0), 1)
    lines = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()); print(lines)
    if write:
        MACROS = PAPER_DIR / "results_macros.tex"; s = MACROS.read_text(); base = "% ---------- Application, parent-comment design (CMV deep"
        if regressor == "mlp":   # the network comparison is not in the paper: its block carries the infix Nn
            lines = "\n".join(f"\\newcommand{{\\{k.replace('resDeep', 'resDeepNn', 1)}}}{{{v}}}" for k, v in M.items()); head = base + "), NETWORK COMPARISON (app_cmv_deep.py --regressor mlp; not used in the paper) ----------"
            drop = lambda l: re.match(r"\\newcommand\{\\resDeepNn", l) is not None or l.startswith(head)
        else:
            head = base + ") ----------"; drop = lambda l: re.match(r"\\newcommand\{\\resDeep(?!Nn)", l) is not None or (l.startswith(base) and "NETWORK" not in l)
        s = "\n".join(l for l in s.splitlines() if not drop(l)).rstrip("\n")
        MACROS.write_text(s + "\n" + head + "\n" + lines + "\n"); print("written to", MACROS)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="build,embed,rate,agree,test,components,plots,macros")
    ap.add_argument("--budget", type=float, default=20.0); ap.add_argument("--regressor", default="ols"); ap.add_argument("--write", action="store_true")
    ap.add_argument("--rater", default="openai", choices=list(RATERS)); ap.add_argument("--smoke", action="store_true", help="networks: the first pair with E1, one seed and one configuration, into *_smoke files")
    ap.add_argument("--views", default=None, help="networks: comma-separated embeddings to fit for every pair (default: all for the two score pairs, E1 for the others)"); ap.add_argument("--no-bench", action="store_true", help="networks: skip the benchmarks")
    a = ap.parse_args(); ph = a.only.split(","); sfx = fsfx = "_nn" if a.regressor == "mlp" else ""
    if "build" in ph: phase_build()
    if "embed" in ph: phase_embed()
    if "rate" in ph: phase_rate(a.rater, a.budget)
    if "agree" in ph: phase_agree(a.rater)
    if "test" in ph: phase_test(a.rater, a.regressor, a.smoke, a.views.split(",") if a.views else None, not a.no_bench)
    if "components" in ph: phase_components(a.rater, sfx)
    if "plots" in ph: phase_plots(a.rater, sfx, fsfx)
    if "macros" in ph: phase_macros(a.rater, a.write, sfx, a.regressor)
    log("ALL DONE")
