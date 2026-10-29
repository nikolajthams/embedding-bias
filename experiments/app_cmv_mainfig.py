"""Main-text figure for the application: for each of the six (X, Y) pairs, the stratified reference rho_id (conditioning
on the post itself) against the embedded rho_psi with psi = E1, and the marginal correlation (no conditioning), all with 95% intervals. The eight-view version of every
pair is app_cmv_appfig.py (appendix). Output: paper/figures/app_cmv_main.pdf. Usage: python experiments/app_cmv_mainfig.py"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from citext import plotting as P; from citext.paths import RESULTS_DIR
res = pd.read_csv(RESULTS_DIR / "app_cmv_results.csv")
pairs = [("politeness", "score"), ("politeness", "delta"), ("log_length", "score"), ("log_length", "delta"), ("lexicon", "score"), ("lexicon", "delta")]
xn = {"politeness": "politeness", "log_length": "log length", "lexicon": "lexicon"}
labels = [f"{xn[x]}\n{y}" for x, y in pairs]
series = [("marginal (no conditioning)", r"no conditioning, $\hat\rho_0$", P.MUTED, "D", -0.22),
          ("stratified (OP identity)", r"stratified reference $\hat\rho_{\mathrm{id}}$", P.CAT[2], "s", 0.0),
          ("E1", r"embedded test, $\psi = E_1$", P.CAT[0], "o", 0.22)]
P.setup(); fig, ax = plt.subplots(figsize=P.SINGLE); xs = np.arange(len(pairs))
for psi, name, col, mk, dx in series:
    rows = [res[(res.X == x) & (res.Y == y) & (res.psi == psi)].iloc[0] for x, y in pairs]
    ax.errorbar(xs + dx, [r.rho for r in rows], yerr=1.96 * np.array([r.se for r in rows]), fmt=mk, ms=3.8, color=col, capsize=2, lw=1, label=name)
ax.axhline(0, color=P.AXIS, lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.3); ax.set_ylabel(r"$\hat\rho$ with 95% interval")
ax.legend(loc="upper left", fontsize=6.3)
P.savefig(fig, "app_cmv_main.pdf"); print("app_cmv_main.pdf written")
