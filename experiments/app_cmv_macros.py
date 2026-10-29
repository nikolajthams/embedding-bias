"""Print the \\resApp* macros for paper/results_macros.tex from the application result files.
Usage: python experiments/app_cmv_macros.py [--write]   (--write replaces the Application block in results_macros.tex)
"""
import sys, json, re; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from citext.paths import RESULTS_DIR, CACHE_DIR, PAPER_DIR
MACROS = PAPER_DIR / "results_macros.tex"
res = pd.read_csv(RESULTS_DIR / "app_cmv_results.csv"); ben = pd.read_csv(RESULTS_DIR / "app_cmv_bench.csv")
comp = json.load(open(RESULTS_DIR / "app_cmv_components.json")); usage = json.load(open(RESULTS_DIR / "app_cmv_rating_usage.json"))
STRAT, MARG = "stratified (OP identity)", "marginal (no conditioning)"
def sub(x, y): return res[(res.X == x) & (res.Y == y)].set_index("psi")
def emb(x, y): r = sub(x, y); return r[~r.index.isin([STRAT, MARG])]
PAIRS = [(x, y) for x in ("politeness", "log_length") for y in ("score", "delta")]
marg_diff_max = max(abs(sub(x, y).loc[MARG, "rho"] - sub(x, y).loc[STRAT, "rho"]) for x, y in PAIRS) if MARG in res.psi.values else float("nan")
def f3(v): return f"{v:.3f}"
def rng(x, y, sep): e = emb(x, y).rho; return f"{f3(e.min())} {sep} {f3(e.max())}"
def pct(v, nd=0): return (f"{100*v:.{nd}f}" if nd else f"{100*v:.0f}") + "\\%"
def thousands(n): return f"{n:,}".replace(",", "{,}")
ps = sub("politeness", "score"); st = ps.loc[STRAT]
df = pd.read_parquet(CACHE_DIR / "cmv" / "rows.parquet")
rat = pd.DataFrame([json.loads(l) for l in (CACHE_DIR / "cmv" / "ratings_haiku.jsonl").read_text().splitlines() if l.strip()]).dropna(subset=["rating"]).drop_duplicates("id")
rated = df[df.id.isin(rat.id)]
bias_max = max(abs(emb(x, y).rho - sub(x, y).loc[STRAT, "rho"]).max() for x in ("politeness", "log_length") for y in ("score", "delta"))
c = comp["politeness/score"]
perm_p = st.perm_p; perm_txt = "0.001" if perm_p <= 0.0011 else f"{perm_p:.3f}"
M = {
 "resAppNops": thousands(df.op.nunique()), "resAppN": thousands(len(df)),
 "resAppStratRho": f3(st.rho), "resAppStratP": perm_txt, "resAppRated": thousands(len(rated)), "resAppStratSe": f3(st.se),
 "resAppEmbRange": rng("politeness", "score", "and"),
 "resAppDeltaStrat": f3(sub("politeness", "delta").loc[STRAT, "rho"]), "resAppDeltaEmbRange": rng("politeness", "delta", "to"),
 "resAppLenStrat": f3(sub("log_length", "score").loc[STRAT, "rho"]), "resAppLenEmbRange": rng("log_length", "score", "to"),
 "resAppLenDeltaStrat": f3(sub("log_length", "delta").loc[STRAT, "rho"]), "resAppLenDeltaEmbRange": rng("log_length", "delta", "to"),
 "resAppBiasMax": f3(bias_max),
 "resAppShareX": pct(c["share_between_X"]), "resAppShareY": pct(c["share_between_Y"]),
 "resAppEmbRtwoX": pct(max(c["emb_R2_X_E1"], 0)), "resAppEmbRtwoY": pct(max(c["emb_R2_Y_E1"], 0), 1),
 "resAppBenchMax": f3(ben.rho_bench.max()), "resAppLambda": f"{c['lambda_hat']:.2f}",
 "resAppPredBias": f3(c["predicted_bias_E1_recursion"]), "resAppMeasBias": f3(c["measured_bias_E1"]),
 "resAppPredBiasCov": f3(c["predicted_bias_E1"]), "resAppPredRho": f3(c["predicted_rho_E1_recursion"]), "resAppRhoEone": f3(c["rho_E1"]),
 "resAppRtwoA": f3(c["R2_A_E1_def"]), "resAppRtwoB": f3(c["R2_B_E1_def"]), "resAppBoundEone": f3(c["bound_E1_def"]), "resAppShrink": f"{c['shrink_E1']:.2f}",
 "resAppRValpha": f"{ps.loc['E1', 'RV_alpha']:.2f}", "resAppRatingCost": f"{usage['cost_usd']:.1f}",
 "resAppRatedPosts": thousands(rated.op.nunique()),
 "resAppMargPolScore": f3(ps.loc[MARG, "rho"]) if MARG in ps.index else "TBD", "resAppMargDiffMax": f3(marg_diff_max),
}
lines = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items())
print(lines)
if "--write" in sys.argv:
    s = MACROS.read_text(); i = s.index("% ---------- Application (CMV) ----------")
    head = s[:i] + "% ---------- Application (CMV) ----------\n"; tail = s[i:]
    rest = "\n".join(l for l in tail.splitlines()[1:] if not re.match(r"\\newcommand\{\\resApp", l))
    MACROS.write_text(head + lines + ("\n" + rest if rest.strip() else "\n")); print("written to", MACROS)
