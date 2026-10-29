"""Appendix figure for the application: all six (X, Y) pairs, embedded rho_hat per view against the stratified reference,
with the insufficiency of E1 demonstrated by each added view.  Output: paper/figures/app_cmv_all_pairs.pdf"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from citext import plotting as P; from citext.paths import RESULTS_DIR
res = pd.read_csv(RESULTS_DIR / "app_cmv_results.csv"); ben = pd.read_csv(RESULTS_DIR / "app_cmv_bench.csv")
order = ["E1", "E2", "E1c", "E2c", "[E1,S]", "[E1,L300]", "prognostic", "[E1,prognostic]"]
labels = {"E1": "$E_1$", "E2": "$E_2$", "E1c": "$E_1^c$", "E2c": "$E_2^c$", "[E1,S]": "$[E_1,S]$", "[E1,L300]": "$[E_1,L]$", "prognostic": "prog.", "[E1,prognostic]": "$[E_1,$pr.$]$"}
view_of = {"E2": "E2", "E1c": "E1c", "E2c": "E2c", "[E1,S]": "S", "[E1,L300]": "L300", "prognostic": "prognostic", "[E1,prognostic]": "prognostic"}
xnames = {"politeness": "politeness", "log_length": "log length", "lexicon": "lexicon"}
P.setup(); fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.2), sharex=True)
for j, xn in enumerate(["politeness", "log_length", "lexicon"]):
    for i, yn in enumerate(["score", "delta"]):
        ax = axs[i, j]; r = res[(res.X == xn) & (res.Y == yn)].set_index("psi"); b = ben[(ben.X == xn) & (ben.Y == yn)].set_index("view")
        if "stratified (OP identity)" not in r.index: ax.set_visible(False); continue
        st = r.loc["stratified (OP identity)"]
        ax.axhspan(st.rho - 1.96 * st.se, st.rho + 1.96 * st.se, color=P.CAT[2], alpha=0.18, lw=0); ax.axhline(st.rho, color=P.CAT[2], lw=1.2, label=r"stratified $\hat\rho_{\mathrm{id}}$, 95% band")
        xs = np.arange(len(order)); rh = [r.loc[k, "rho"] for k in order]; se = [r.loc[k, "se"] for k in order]
        ax.errorbar(xs, rh, yerr=1.96 * np.array(se), fmt="o", ms=3.5, color=P.CAT[0], capsize=1.5, label=r"embedded $\hat\rho_\psi$")
        bx = [k for k, v in enumerate(order) if v in view_of]; bv = [b.loc[view_of[order[k]], "rho_bench"] for k in bx]
        ax.scatter(bx, bv, marker="s", color=P.CAT[1], s=16, zorder=3, label=r"$\hat\rho_{\mathrm{bench}}(E_1\to\psi')$")
        ax.axhline(0, color=P.AXIS, lw=0.6); ax.set_xticks(xs); ax.set_xticklabels([labels[k] for k in order], fontsize=5.6, rotation=90)
        ax.set_title(f"{xnames[xn]} against {yn}", fontsize=7.5)
        if j == 0: ax.set_ylabel(r"$\hat\rho$")
axs[0, 0].legend(loc="lower left", fontsize=5.8); fig.subplots_adjust(hspace=0.35)
P.savefig(fig, "app_cmv_all_pairs.pdf"); print("app_cmv_all_pairs.pdf written")
