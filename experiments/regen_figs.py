"""NOTE (7 Oct 2026): the Experiment 3 functions are OBSOLETE (Experiment 3 was removed from the paper).
Regenerate figures from the saved CSV/JSON results (no recomputation). Labels follow the paper's notation
(X, Y tested variables; rho_psi; tolerance B; M regenerated comments).
  exp2_text_typeI.pdf             from results/exp2_text_typeI.csv           (legend moved off the curves)
  exp2_text_power.pdf             from results/exp2_text_power.csv
  exp3_llm_learning_curve.pdf     from results/exp3_llm_subsample_raw.csv, exp3_llm_full_sample.csv, summary_exp3.json
  exp3_llm_learning_curve_h0.pdf  the H0 panel of the learning curve alone (main text)
  exp3_llm_crt.pdf                from results/exp3_diagnostics.json (draws of the model-X reference)
Usage: python experiments/regen_figs.py
"""
import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from citext import plotting as P; from citext.paths import RESULTS_DIR
ALPHA = 0.05; NS = [500, 1000, 2000, 4000]; REG = "ridge"   # exp3 primary regressor (ridge throughout since 30 Sept 2026)
COLS = {"E1": P.CAT[0], "E2": P.CAT[1], "E1+E2": P.CAT[2]}

def log_axis(ax, ns):
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off(); ax.set_xlabel(r"sample size $n$")

def exp2_typeI():
    agg = pd.read_csv(RESULTS_DIR / "exp2_text_typeI.csv"); s2 = json.loads((RESULTS_DIR / "summary_exp2.json").read_text())
    reg, ns = s2["primary_regressor"], sorted(agg.n.unique())
    P.setup(); fig, ax = plt.subplots(figsize=P.SINGLE)
    series = [(("E1", 1.0), r"$\psi=E_1$ (insufficient), $\lambda=1$", P.seq_colors(2)[1], dict(marker="o")),
              (("E1", 0.0), r"$\psi=E_1$ (insufficient), $\lambda=0$", P.seq_colors(2)[0], dict(marker="o", linestyle=":", markerfacecolor="white", markeredgewidth=1.2)),
              (("rich", 1.0), r"$\psi=[E_1,E_2]$ (sufficient, %d-dim)" % s2["rich_encoder_dim"], P.CAT[1], dict(marker="o")),
              (("oracle", 1.0), r"$\psi=$ generating features (sufficient)", P.CAT[2], dict(marker="s", linestyle="--"))]
    for (psi, lam), lab, c, st in series:
        a = agg[(agg.psi == psi) & (agg.lam == lam)].sort_values("n"); kw = dict(color=c, label=lab, **st)
        if "markerfacecolor" in st: kw["markeredgecolor"] = c
        ax.plot(a.n, a[f"rej_{reg}"], **kw)
    ax.axhline(ALPHA, color=P.MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate(r"$\alpha=0.05$", (ns[-1], ALPHA), xytext=(-2, 3), textcoords="offset points", fontsize=6.5, color=P.INK2, ha="right")
    log_axis(ax, ns); ax.set_ylabel("rejection rate under $H_0$"); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center right", fontsize=6.2, bbox_to_anchor=(1.0, 0.45))
    P.savefig(fig, "exp2_text_typeI.pdf"); print("exp2_text_typeI.pdf")

def exp2_power():
    df = pd.read_csv(RESULTS_DIR / "exp2_text_power.csv")
    if "gamma" not in df.columns: df = df.rename(columns={"beta": "gamma"})
    P.setup(); fig, axs = plt.subplots(1, 2, figsize=P.DOUBLE, sharey=True)
    titles = {0.0: r"$\lambda=0$: insufficient $\psi$, no bias", 1.0: r"$\lambda=1$: insufficient $\psi$, bias at the bound"}
    for ax, lam in zip(axs, (0.0, 1.0)):
        d = df[df.lam == lam].sort_values("gamma")
        ax.plot(d.gamma, d.naive, marker="o", color=P.CAT[1], label="standard test")
        ax.plot(d.gamma, d.robust_bound, marker="o", color=P.CAT[6], label=r"robust, $B=$ oracle bound")
        ax.axhline(ALPHA, color=P.MUTED, ls="--", lw=0.8); ax.set_title(titles[lam])
        ax.set_xlabel(r"effect size $\gamma$ ($Y \leftarrow Y + \gamma X$)"); ax.set_ylim(-0.02, 1.02)
    axs[0].set_ylabel("rejection rate"); axs[0].legend(loc="center right")
    P.savefig(fig, "exp2_text_power.pdf"); print("exp2_text_power.pdf")

def _exp3_curves():
    raw = pd.read_csv(RESULTS_DIR / "exp3_llm_subsample_raw.csv"); full = pd.read_csv(RESULTS_DIR / "exp3_llm_full_sample.csv")
    marg = json.loads((RESULTS_DIR / "summary_exp3.json").read_text())["sentiment_following"]
    raw = raw[(raw.axis == "sentiment") & (raw.reg == REG)]
    agg = raw.groupby(["hyp", "psi", "n"]).rho.agg(med="median", lo=lambda s: s.quantile(.25), hi=lambda s: s.quantile(.75)).reset_index()
    return agg, full, marg

def _draw_curve(ax, agg, full, hyp, psi, label):
    """Median and interquartile range over subsamples, plus the full-sample value at n = 3000 with a 95% interval."""
    a = agg[(agg.hyp == hyp) & (agg.psi == psi)].sort_values("n")
    f = full[(full.axis == "sentiment") & (full.hyp == hyp) & (full.psi == psi) & (full.reg == REG)]
    ns = list(a.n) + [3000]; med = list(a.med) + [float(f.rho_hat.iloc[0])]
    lo = list(a.lo) + [float(f.rho_hat.iloc[0] - 1.96 * f.se.iloc[0])]; hi = list(a.hi) + [float(f.rho_hat.iloc[0] + 1.96 * f.se.iloc[0])]
    ax.fill_between(ns, lo, hi, color=COLS[psi], alpha=0.15, linewidth=0); ax.plot(ns, med, marker="o", color=COLS[psi], label=label)

def exp3_learning_curve():
    agg, full, marg = _exp3_curves()
    P.setup(); fig, axs = plt.subplots(1, 2, figsize=P.DOUBLE, sharey=True)
    labs = {"E1": r"$\psi=E_1(Z)$", "E2": r"$\psi=E_2(Z)$", "E1+E2": r"$\psi=[E_1,E_2](Z)$"}
    for ax, hyp, title, m in zip(axs, ("H0", "H1"), (r"$H_0$: $Y$ generated from $Z$ alone", r"$H_1$: $Y$ matches the sentiment of $X$"),
                                 (marg["corr_A_B0"], marg["corr_A_B1"])):
        for psi in ("E1", "E2", "E1+E2"):
            _draw_curve(ax, agg, full, hyp, psi, labs[psi])
        ax.axhline(m, color=P.MUTED, ls=":", lw=0.9); ax.annotate("unadjusted correlation of $X$ and $Y$", (500, m), xytext=(2, 3), textcoords="offset points", fontsize=6.5, color=P.INK2)
        ax.axhline(0, color=P.AXIS, lw=0.6); log_axis(ax, [500, 1000, 2000, 3000]); ax.set_title(title)
    axs[0].set_ylabel(r"residual correlation $\hat\rho$"); axs[0].legend(loc="lower left")
    P.savefig(fig, "exp3_llm_learning_curve.pdf"); print("exp3_llm_learning_curve.pdf")

def exp3_learning_curve_h0():
    agg, full, marg = _exp3_curves(); m = float(marg["corr_A_B0"])
    P.setup(); fig, ax = plt.subplots(figsize=P.SINGLE)
    labs = {"E1": r"$\psi=E_1$", "E2": r"$\psi=E_2$", "E1+E2": r"$\psi=[E_1,E_2]$"}
    for psi in ("E1", "E2", "E1+E2"):
        _draw_curve(ax, agg, full, "H0", psi, labs[psi])
    ax.axhline(m, color=P.MUTED, ls=":", lw=0.9)
    ax.annotate("unadjusted correlation of $X$ and $Y$", (500, m), xytext=(2, 3), textcoords="offset points", fontsize=6.5, color=P.INK2)
    ax.axhline(0, color=P.AXIS, lw=0.6); log_axis(ax, [500, 1000, 2000, 3000])
    ax.set_ylabel(r"$\hat\rho$ under $H_0$"); ax.set_ylim(0, m + 0.05); ax.legend(loc="lower left")
    P.savefig(fig, "exp3_llm_learning_curve_h0.pdf"); print("exp3_llm_learning_curve_h0.pdf")

def exp3_crt():
    c = json.loads((RESULTS_DIR / "exp3_diagnostics.json").read_text())["crt"]
    rho_b = np.asarray(c["rho_draws"], dtype=float); M = int(c["B"]); obs, se = float(c["rho_obs"]), float(c["se_obs"])
    P.setup(); fig, ax = plt.subplots(figsize=P.SINGLE)
    ax.axvspan(-1.96 * se, 1.96 * se, color=P.CAT[1], alpha=0.15, lw=0)
    ax.annotate("acceptance region of the\nembedded test", (0, 0.04), xytext=(0, 0), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=P.INK2)
    rng = np.random.default_rng(0); yj = rng.uniform(0.15, 0.35, size=M)
    ax.scatter(rho_b, yj, s=18, color=P.CAT[0], label=rf"$\hat\rho$ for regenerated comments ($M={M}$)", zorder=3)
    ax.scatter([obs], [0.25], s=40, marker="D", color=P.CAT[7], label=r"observed $\hat\rho$", zorder=4)
    ax.set_yticks([]); ax.set_ylim(0, 0.6); ax.set_xlabel(r"residual correlation $\hat\rho$ ($\psi=E_1$, $n'=%d$)" % int(c["n_sub"]))
    ax.legend(loc="upper right"); ax.set_xlim(min(-0.1, rho_b.min() - 0.03), max(rho_b.max(), obs) + 0.05)
    P.savefig(fig, "exp3_llm_crt.pdf"); print("exp3_llm_crt.pdf")

if __name__ == "__main__":
    exp2_typeI(); exp2_power(); exp3_learning_curve(); exp3_learning_curve_h0(); exp3_crt()
