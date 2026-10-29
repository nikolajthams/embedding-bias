"""Application: style and success of replies on r/ChangeMyView (ConvoKit winning-args-corpus).
Phases (all resumable, results in code/results/app_cmv_*):
  build  -> rows (top-level replies, <= 8 per OP, >= 20 words), OP table, lexicon features, delta labels
  embed  -> OP views: E1, E2 (384 window), chunked E1c/E2c, surface S, TF-IDF SVD L300; cached under ~/.cache/ci_text/cmv
  rate   -> LLM politeness rating (claude-haiku-4-5, temperature 0) with a pilot and a 10 USD guard; cached JSONL
  test   -> embedded tests (ridge primary, GroupKFold by OP, cluster-robust se), benchmarks, stratified reference + permutation
  plots  -> paper/figures/app_cmv_results.pdf
usage: python experiments/app_cmv.py [--only build,embed,rate,test,plots] [--max-per-op 8]"""
from __future__ import annotations
import argparse, asyncio, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from citext import plotting as P
from citext.estimators import fit_predict_crossfit, partial_r2_benchmark, rv_alpha, critical_value
from citext.paths import RESULTS_DIR, CACHE_DIR
from citext.text_semi import E1_NAME, E2_NAME, _device, _safe

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
except Exception:
    pass
CMV = CACHE_DIR / "cmv"; CORPUS = CMV / "winning-args-corpus"
ROWS = CMV / "rows.parquet"; OPS = CMV / "ops.parquet"; RATINGS = CMV / "ratings_haiku.jsonl"
ALPHA, SEED, K = 0.05, 0, 5
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ build
POLITE = [r"\bplease\b", r"\bthank(s| you)\b", r"\bappreciat", r"\bsorry\b", r"\bapolog", r"\bi think\b", r"\bi believe\b", r"\bi feel\b",
          r"\bperhaps\b", r"\bmaybe\b", r"\bmight\b", r"\bcould\b", r"\byou'?re right\b", r"\b(good|fair|great) point\b", r"\brespectfully\b", r"\bwith respect\b", r"\bI (would|'d) argue\b"]
RUDE = [r"\bwrong\b", r"\bstupid\b", r"\bridiculous\b", r"\bnonsense\b", r"\bbullshit\b", r"\bidiot", r"\bdumb\b", r"\bignorant\b", r"\bpathetic\b", r"\bshut up\b"]
def lexicon(text):
    t = text.lower(); w = max(len(t.split()), 1)
    pol = sum(len(re.findall(p, t)) for p in POLITE); rude = sum(len(re.findall(p, t)) for p in RUDE)
    return pol / w * 100, rude / w * 100, t.count("?") / w * 100, t.count("!") / w * 100, sum(c.isupper() for c in text) / max(len(text), 1)

def phase_build(max_per_op: int):
    convs = json.load(open(CORPUS / "conversations.json"))
    ops = pd.DataFrame([{"op": cid, "title": m.get("op-title") or "", "body": m.get("op-text-body") or ""} for cid, m in convs.items()])
    ops["text"] = (ops.title + "\n\n" + ops.body).str.replace(r"\s+", " ", regex=True).str.strip()
    parent, root_of, user_of, rows, delta_bot = {}, {}, {}, [], []
    with open(CORPUS / "utterances.jsonl") as f:
        for line in f:
            u = json.loads(line); rt = u.get("reply-to"); root = u.get("root"); parent[u["id"]] = rt; root_of[u["id"]] = root; user_of[u["id"]] = u.get("user")
            if u.get("user") == "DeltaBot":
                m = re.search(r"delta awarded to /u/([\w-]+)", u.get("text") or ""); delta_bot.append((u["id"], rt, root, m.group(1) if m else None))
            elif rt is not None and rt == root:
                text = (u.get("text") or "").strip()
                if text in ("[deleted]", "[removed]") or u.get("user") in (None, "[deleted]"): continue
                rows.append({"op": root, "id": u["id"], "user": u["user"], "ts": u.get("timestamp"), "text": re.sub(r"\s+", " ", text), "score": u["meta"].get("score"), "success_pair": u["meta"].get("success")})
    df = pd.DataFrame(rows); log(f"{len(df)} top-level replies parsed, {len(delta_bot)} DeltaBot confirmations")
    top = {}
    for _, rt, root, user in delta_bot:               # walk up from the comment DeltaBot replied to (the OP's delta comment) to the top-level reply
        cur = rt; steps = 0
        while cur is not None and parent.get(cur) is not None and parent.get(cur) != root_of.get(cur) and steps < 100:
            cur = parent.get(cur); steps += 1
        if cur is not None and parent.get(cur) == root_of.get(cur):
            top[(root, cur)] = 1
        elif user is not None:
            top[(root, "user:" + user)] = 1
    df["delta"] = [int((r.op, r.id) in top or (r.op, "user:" + str(r.user)) in top) for r in df.itertuples()]
    df["n_words"] = df.text.str.split().str.len(); df = df[(df.n_words >= 20) & df.score.notna()].copy()
    df["score"] = df.score.astype(float); df["y_score"] = np.sign(df.score) * np.log1p(np.abs(df.score)); df["log_len"] = np.log(df.n_words)
    lex = np.array([lexicon(t) for t in df.text]); df["lex_polite"], df["lex_rude"], df["lex_q"], df["lex_excl"], df["lex_upper"] = lex.T
    df["x_lex"] = df.lex_polite - df.lex_rude
    rng = np.random.default_rng(SEED); df = df.sample(frac=1.0, random_state=SEED)
    df = df.groupby("op", group_keys=False).head(max_per_op).sort_values(["op", "ts"]).reset_index(drop=True)
    n_ops = df.op.nunique(); df = df[df.groupby("op").op.transform("size") >= 3].reset_index(drop=True)
    log(f"rows kept: {len(df)} from {df.op.nunique()} OPs (of {n_ops}); delta rate {df.delta.mean():.3f}; median score {df.score.median()}")
    ops = ops[ops.op.isin(df.op)].reset_index(drop=True); ops["n_words"] = ops.text.str.split().str.len()
    log(f"OP words: median {ops.n_words.median()}, 90% {ops.n_words.quantile(.9):.0f}")
    df.to_parquet(ROWS); ops.to_parquet(OPS)
    json.dump({"n_rows": int(len(df)), "n_ops": int(df.op.nunique()), "delta_rate": float(df.delta.mean()), "n_delta": int(df.delta.sum()),
               "n_top_level_all": int(len(rows)), "n_deltabot": len(delta_bot), "median_score": float(df.score.median()),
               "op_words_median": float(ops.n_words.median()), "op_words_q90": float(ops.n_words.quantile(.9)), "max_per_op": max_per_op},
              open(RESULTS_DIR / "app_cmv_build.json", "w"), indent=1)

# ------------------------------------------------------------------ embed
def embed_len(texts, model_name, max_len, key, chunk_words=None):
    f = CMV / f"emb_{key}_{_safe(model_name)}_{max_len}_{chunk_words or 0}.npy"
    if f.exists(): return np.load(f)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, cache_folder=str(CACHE_DIR / "st_models"), device=_device()); model.max_seq_length = max_len
    if chunk_words is None:
        E = model.encode(list(texts), batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    else:
        chunks, owner, weight = [], [], []
        for i, t in enumerate(texts):
            w = t.split(" ")
            for j in range(0, len(w), chunk_words): c = w[j:j + chunk_words]; chunks.append(" ".join(c)); owner.append(i); weight.append(len(c))
        Ec = model.encode(chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True); owner = np.array(owner); weight = np.array(weight, float)
        E = np.zeros((len(texts), Ec.shape[1])); np.add.at(E, owner, Ec * weight[:, None]); E /= np.bincount(owner, weights=weight, minlength=len(texts))[:, None]
    E = np.asarray(E, dtype=np.float64); np.save(f, E); return E

def surface(texts):
    F = []
    for t in texts:
        w = t.split(); nc = max(len(t), 1)
        F.append([np.log1p(len(w)), np.log1p(len(t)), len(re.findall(r"[.!?]+", t)), sum(c.isupper() for c in t) / nc, sum(c.isdigit() for c in t) / nc,
                  float(bool(re.search(r"https?://|www\.", t))), np.mean([len(x) for x in w]) if w else 0.0, t.count("?") / nc * 100])
    return np.array(F)

def phase_embed():
    ops = pd.read_parquet(OPS); texts = list(ops.text)
    views = {"E1": embed_len(texts, E1_NAME, 256, "op"), "E2": embed_len(texts, E2_NAME, 384, "op"),
             "E1c": embed_len(texts, E1_NAME, 256, "op", chunk_words=150), "E2c": embed_len(texts, E2_NAME, 384, "op", chunk_words=200), "S": surface(texts)}
    Tf = TfidfVectorizer(min_df=2, max_df=0.9, sublinear_tf=True, max_features=40000).fit_transform(texts)
    views["L300"] = TruncatedSVD(300, random_state=0).fit_transform(Tf)
    np.savez(CMV / "views.npz", **views); log("views:", {k: v.shape for k, v in views.items()})
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(E1_NAME, cache_folder=str(CACHE_DIR / "st_models"), device="cpu")
    L = np.array([len(x) for x in m.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]])
    json.dump({"frac_over_256": float((L > 256).mean()), "frac_over_384": float((L > 384).mean()), "median_tokens": float(np.median(L))}, open(RESULTS_DIR / "app_cmv_truncation.json", "w"))
    log(f"OP tokens: median {np.median(L):.0f}, frac>256 {(L>256).mean():.2f}, frac>384 {(L>384).mean():.2f}")

# ------------------------------------------------------------------ rate
SYSTEM = "You rate the politeness of online comments for a research dataset. Judge tone and courtesy only, not the opinion expressed. Answer with a single digit."
PROMPT = "Comment:\n<<<\n{text}\n>>>\n\nRate the politeness of this comment from 1 (rude or hostile) to 5 (very polite and courteous). Answer with a single digit."
PRICE_IN, PRICE_OUT = 1.0, 5.0  # USD per 1M tokens, claude-haiku-4-5
def phase_rate(model: str, budget_usd: float):
    import anthropic
    df = pd.read_parquet(ROWS); done = {}
    if RATINGS.exists():
        for l in RATINGS.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("rating") is not None: done[r["id"]] = r   # failed calls (rating None) are retried
    todo = df[~df.id.isin(done)][["id", "text"]].values.tolist(); log(f"ratings: {len(done)} cached, {len(todo)} to do")
    if not todo: return
    tot = {"in": 0, "out": 0}
    async def one(client, sem, rid, text):
        async with sem:
            for attempt in range(3):
                try:
                    kw = dict(model=model, max_tokens=5, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                              messages=[{"role": "user", "content": PROMPT.format(text=text[:6000])}])
                    try:
                        r = await client.messages.create(extra_body={"temperature": 0.0}, **kw)
                    except anthropic.BadRequestError:
                        r = await client.messages.create(**kw)
                    out = "".join(b.text for b in r.content if b.type == "text"); m = re.search(r"[1-5]", out)
                    tot["in"] += r.usage.input_tokens + (r.usage.cache_read_input_tokens or 0) // 10; tot["out"] += r.usage.output_tokens
                    return {"id": rid, "rating": int(m.group()) if m else None, "raw": out[:10]}
                except Exception as e:
                    if attempt == 2: return {"id": rid, "rating": None, "raw": f"ERR {type(e).__name__}: {str(e)[:60]}"}
                    await asyncio.sleep(2 * (attempt + 1))
    async def run(items, conc):
        client = anthropic.AsyncAnthropic(); sem = asyncio.Semaphore(conc)
        with open(RATINGS, "a") as f:
            for k in range(0, len(items), 400):
                res = await asyncio.gather(*[one(client, sem, rid, text) for rid, text in items[k:k + 400]])
                for r in res: f.write(json.dumps(r) + "\n")
                cost = tot["in"] / 1e6 * PRICE_IN + tot["out"] / 1e6 * PRICE_OUT
                log(f"  rated {min(k+400, len(items))}/{len(items)}, running cost {cost:.2f} USD")
    pilot = todo[:20]; asyncio.run(run(pilot, 8))
    per_call = (tot["in"] / 1e6 * PRICE_IN + tot["out"] / 1e6 * PRICE_OUT) / max(len(pilot), 1); est = per_call * len(todo)
    log(f"pilot: {per_call*1000:.3f} USD per 1000 calls; extrapolated {est:.2f} USD for {len(todo)} calls")
    if est > budget_usd:
        log(f"ABORT rating: estimate {est:.2f} USD exceeds budget {budget_usd} USD"); return
    asyncio.run(run(todo[20:], 16))
    cost = tot["in"] / 1e6 * PRICE_IN + tot["out"] / 1e6 * PRICE_OUT
    usage_path = RESULTS_DIR / "app_cmv_rating_usage.json"; prev = json.load(open(usage_path)) if usage_path.exists() else {}
    n_ok = sum(1 for l in RATINGS.read_text().splitlines() if l.strip() and json.loads(l).get("rating") is not None)
    json.dump({"model": model, "n_calls": prev.get("n_calls", 0) + len(todo), "n_rated": n_ok, "tokens": {k: prev.get("tokens", {}).get(k, 0) + v for k, v in tot.items()},
               "cost_usd": prev.get("cost_usd", 0.0) + cost, "cost_usd_this_run": cost, "temperature": 0.0}, open(usage_path, "w"))
    log(f"rating done: {cost:.2f} USD")

# ------------------------------------------------------------------ test
def cluster_se(IF, groups):
    s = pd.Series(IF).groupby(groups).sum().values; return float(np.sqrt(np.sum(s**2)) / len(IF))
def corr_stats(rA, rB, groups, alpha=ALPHA):
    a = rA / np.sqrt(np.mean(rA**2)); b = rB / np.sqrt(np.mean(rB**2)); rho = float(np.mean(a * b))
    IF = a * b - rho / 2 * (a**2 + b**2); se = cluster_se(IF, groups)
    from scipy.stats import norm
    p = float(2 * norm.sf(abs(rho) / se)); return {"rho": rho, "se": se, "p": p, "RV": abs(rho), "RV_alpha": float(rv_alpha(rho, se, alpha)), "reject": bool(abs(rho) > critical_value(0.0, se, alpha))}
def within_op(x, groups):
    return x - pd.Series(x).groupby(groups).transform("mean").values
def stratified(X, Y, groups, B=1000, seed=SEED):
    xt, yt = within_op(X, groups), within_op(Y, groups); st = corr_stats(xt, yt, groups)
    rng = np.random.default_rng(seed); g = np.asarray(groups); order = np.argsort(g, kind="stable"); gs = g[order]; Xs = X[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True]); rho_b = []
    denom = np.sqrt(np.sum(xt**2) * np.sum(yt**2))
    for _ in range(B):
        Xp = Xs.copy()
        for i in range(len(bounds) - 1):
            seg = slice(bounds[i], bounds[i + 1]); Xp[seg] = Xp[seg][rng.permutation(bounds[i + 1] - bounds[i])]
        Xperm = np.empty_like(X); Xperm[order] = Xp; xtp = within_op(Xperm, groups); rho_b.append(np.sum(xtp * yt) / denom)
    rho_b = np.array(rho_b); st["perm_p"] = float((1 + np.sum(np.abs(rho_b) >= abs(st["rho"]))) / (B + 1)); st["perm_sd"] = float(rho_b.std(ddof=1)); return st

def phase_test(regressor="hybrid"):
    df = pd.read_parquet(ROWS); ops = pd.read_parquet(OPS); V = dict(np.load(CMV / "views.npz"))
    if RATINGS.exists():
        rat = pd.DataFrame([json.loads(l) for l in RATINGS.read_text().splitlines() if l.strip()]); rat = rat.dropna(subset=["rating"]).drop_duplicates("id")
        df = df.merge(rat[["id", "rating"]], on="id", how="left"); log(f"ratings merged: {df.rating.notna().sum()} of {len(df)}")
    else:
        df["rating"] = np.nan
    op_index = {o: i for i, o in enumerate(ops.op)}; idx = np.array([op_index[o] for o in df.op]); groups = df.op.values
    folds = list(GroupKFold(K).split(np.zeros(len(df)), groups=groups))
    Xs = {"politeness": df.rating.values.astype(float), "lexicon": df.x_lex.values.astype(float), "log_length": df.log_len.values.astype(float)}
    Ys = {"score": df.y_score.values.astype(float), "delta": df.delta.values.astype(float)}
    Dense = np.hstack([V["L300"], V["S"]])[idx]   # prognostic regressions use the dense lexical + surface view of the post
    results, bench = [], []
    for xn, X in Xs.items():
        ok = ~np.isnan(X)
        if ok.sum() < 100: log(f"skip {xn}: too few values"); continue
        for yn, Y in Ys.items():
            m = ok & ~np.isnan(Y); Xm, Ym, gm = X[m], Y[m], groups[m]; fm = list(GroupKFold(K).split(np.zeros(m.sum()), groups=gm)); im = idx[m]
            strat = stratified(Xm, Ym, gm); results.append({"X": xn, "Y": yn, "psi": "stratified (OP identity)", "n": int(m.sum()), **strat}); log(f"{xn}/{yn} stratified: rho={strat['rho']:.4f} se={strat['se']:.4f} perm_p={strat['perm_p']:.3f}")
            marg = corr_stats(Xm - Xm.mean(), Ym - Ym.mean(), gm)   # conditioning on nothing: the marginal correlation, clustered se
            results.append({"X": xn, "Y": yn, "psi": "marginal (no conditioning)", "n": int(m.sum()), **marg}); log(f"{xn}/{yn} marginal: rho={marg['rho']:.4f} se={marg['se']:.4f}")
            Dm = Dense[m]; predX = np.empty(m.sum()); predY = np.empty(m.sum())
            for tr, te in fm:
                predX[te] = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Dm[tr], Xm[tr]).predict(Dm[te])
                predY[te] = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Dm[tr], Ym[tr]).predict(Dm[te])
            prog = np.c_[predX, predY]
            psis = {"E1": V["E1"][im], "E2": V["E2"][im], "E1c": V["E1c"][im], "E2c": V["E2c"][im], "[E1,S]": np.hstack([V["E1"][im], V["S"][im]]),
                    "[E1,L300]": np.hstack([V["E1"][im], V["L300"][im]]), "prognostic": prog, "[E1,prognostic]": np.hstack([V["E1"][im], prog])}
            for pn, Psi in psis.items():
                rA = Xm - fit_predict_crossfit(Psi, Xm, K, SEED, regressor, fm); rB = Ym - fit_predict_crossfit(Psi, Ym, K, SEED, regressor, fm)
                st = corr_stats(rA, rB, gm); st.update({"R2_X": float(1 - np.mean(rA**2) / np.var(Xm)), "R2_Y": float(1 - np.mean(rB**2) / np.var(Ym))})
                results.append({"X": xn, "Y": yn, "psi": pn, "n": int(m.sum()), **st}); log(f"{xn}/{yn} psi={pn}: rho={st['rho']:.4f} se={st['se']:.4f} RVa={st['RV_alpha']:.4f} R2X={st['R2_X']:.3f} R2Y={st['R2_Y']:.3f}")
            for pn in ("E2", "E1c", "E2c", "S", "L300", "prognostic"):
                Pp = prog if pn == "prognostic" else V[pn][im]
                bA = partial_r2_benchmark(Xm, V["E1"][im], Pp, regressor, K, SEED, fm); bB = partial_r2_benchmark(Ym, V["E1"][im], Pp, regressor, K, SEED, fm)
                bench.append({"X": xn, "Y": yn, "view": pn, "r2_X": bA["r2"], "r2_Y": bB["r2"], "rho_bench": float(np.sqrt(bA["r2"] * bB["r2"]))}); log(f"  bench E1->{pn}: r2X={bA['r2']:.4f} r2Y={bB['r2']:.4f} rho_bench={np.sqrt(bA['r2']*bB['r2']):.4f}")
            pd.DataFrame(results).to_csv(RESULTS_DIR / "app_cmv_results.csv", index=False); pd.DataFrame(bench).to_csv(RESULTS_DIR / "app_cmv_bench.csv", index=False)
    log("tests done")

# ------------------------------------------------------------------ plots
def phase_plots():
    res = pd.read_csv(RESULTS_DIR / "app_cmv_results.csv"); ben = pd.read_csv(RESULTS_DIR / "app_cmv_bench.csv")
    P.setup(); pairs = [("politeness", "score"), ("politeness", "delta")]
    pairs = [p for p in pairs if ((res.X == p[0]) & (res.Y == p[1])).any()] or [("lexicon", "score"), ("lexicon", "delta")]
    fig, axs = plt.subplots(1, len(pairs), figsize=(6.6, 2.4), sharey=False)
    order = ["E1", "E2", "E1c", "E2c", "[E1,S]", "[E1,L300]", "prognostic", "[E1,prognostic]"]
    labels = {"E1": "$E_1$", "E2": "$E_2$", "E1c": "$E_1^c$", "E2c": "$E_2^c$", "[E1,S]": "$[E_1,S]$", "[E1,L300]": "$[E_1,L]$", "prognostic": "prog.", "[E1,prognostic]": "$[E_1,$prog.$]$"}
    view_of = {"E2": "E2", "E1c": "E1c", "E2c": "E2c", "[E1,S]": "S", "[E1,L300]": "L300", "prognostic": "prognostic", "[E1,prognostic]": "prognostic"}
    for ax, (xn, yn) in zip(np.atleast_1d(axs), pairs):
        r = res[(res.X == xn) & (res.Y == yn)].set_index("psi"); b = ben[(ben.X == xn) & (ben.Y == yn)].set_index("view")
        st = r.loc["stratified (OP identity)"]; ax.axhspan(st.rho - 1.96 * st.se, st.rho + 1.96 * st.se, color=P.CAT[2], alpha=0.18, lw=0)
        ax.axhline(st.rho, color=P.CAT[2], lw=1.2, label=r"stratified reference $\hat\rho_{\mathrm{id}}$")
        xs = np.arange(len(order)); rh = [r.loc[k, "rho"] for k in order]; se = [r.loc[k, "se"] for k in order]
        ax.errorbar(xs, rh, yerr=1.96 * np.array(se), fmt="o", color=P.CAT[0], capsize=2, label=r"embedded $\hat\rho_\psi$")
        bx = [i for i, k in enumerate(order) if k in view_of]; bv = [b.loc[view_of[order[i]], "rho_bench"] for i in bx]
        ax.scatter(bx, bv, marker="s", color=P.CAT[1], s=24, zorder=3, label=r"$\hat\rho_{\mathrm{bench}}(E_1 \to \psi')$")
        ax.axhline(0, color=P.AXIS, lw=0.6); ax.set_xticks(xs); ax.set_xticklabels([labels[k] for k in order], fontsize=6.5, rotation=0)
        ax.set_title(f"$X$ = {xn}, $Y$ = {yn}"); ax.set_ylabel(r"$\hat\rho$")
    np.atleast_1d(axs)[0].legend(loc="best", fontsize=6.5); P.savefig(fig, "app_cmv_results.pdf"); log("figure done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="build,embed,rate,test,plots"); ap.add_argument("--max-per-op", type=int, default=8)
    ap.add_argument("--model", default="claude-haiku-4-5"); ap.add_argument("--budget", type=float, default=10.0); ap.add_argument("--regressor", default="hybrid")
    a = ap.parse_args(); ph = a.only.split(",")
    if "build" in ph: phase_build(a.max_per_op)
    if "embed" in ph: phase_embed()
    if "rate" in ph:
        import os
        if os.environ.get("ANTHROPIC_API_KEY"): phase_rate(a.model, a.budget)
        else: log("ANTHROPIC_API_KEY not set: skipping rating (lexicon and length features remain)")
    if "test" in ph: phase_test(a.regressor)
    if "plots" in ph: phase_plots()
    log("ALL DONE")
