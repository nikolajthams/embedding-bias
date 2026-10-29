"""Experiment 3 (local generator) figures from saved results only: the main-text figure (exp4_rejection.pdf: left, rejection rates under H0
for one state view plus E1 and E2; right, rejection rates against the coupled alternatives of exp4_power.py with psi = E1; sized like the
application figure next to which it is placed), the appendix two-panel figure with all four state views (exp4_main.pdf), the appendix power
figure for every embedding (exp4_power_views.pdf), and the appendix identity figure (exp4_identity.pdf).
Regressor (7 Oct 2026): --regressor ols (the default) reads exp4_truth.csv, exp4_power.csv and exp4_local_tests.csv and writes the paper's
figures; --regressor mlp reads the network results exp4_truth_nn.csv, exp4_truth_nn.json and exp4_power_nn.csv and writes the same figures
with the suffix _nn, a comparison that is not in the paper.
Left of exp4_main: rho_hat under H0 against the dimension of psi(Z) on the observed data (the fits of exp4_truth.py; for least squares the
test file of exp4_local.py, the paper's values until 7 Oct 2026) with the true insufficiency B* = sqrt(R2_psi(X) R2_psi(Y)) identified from
the replicates (hollow markers). Right: rejection rates under H0 on the resampled null datasets of the residual correlation test and of the
robust test at the tolerances B*, |lambda| B* and B*/2. Appendix identity figure: |rho_hat| (mean over null datasets) against the prediction
|lambda| B* and the bound B*, one marker per view.
usage: python experiments/fig_exp4.py [--regressor mlp|ols] [--code DIR] [--paper DIR] [--truth CSV] [--power CSV]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--regressor", choices=("ols", "mlp"), default="ols"); ap.add_argument("--code", default=os.environ.get("CI_TEXT_CODE_DIR", str(HERE.parent)))
ap.add_argument("--paper", default=None); ap.add_argument("--truth", default=None); ap.add_argument("--power", default=None); a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from citext import plotting as P
from citext.paths import RESULTS_DIR, PAPER_DIR
PAPER = Path(a.paper) if a.paper else PAPER_DIR; SFX = FSFX = "" if a.regressor == "ols" else "_nn"
TRUTH = Path(a.truth) if a.truth else RESULTS_DIR / f"exp4_truth{SFX}.csv"; POWER = Path(a.power) if a.power else RESULTS_DIR / f"exp4_power{SFX}.csv"
(RESULTS_DIR / "preview").mkdir(exist_ok=True)
def out(name): return PAPER / f"figures/{name}{FSFX}.pdf"
def prev(name): return RESULTS_DIR / f"preview/{name}{FSFX}.png"

T = pd.read_csv(TRUTH); T["fam"] = T.psi.str.split(":").str[0]
if a.regressor == "ols": res = pd.read_csv(CODE / "results/exp4_local_tests.csv"); r0 = res[res.hyp == "H0"].copy()
else:   # the observed data as analysed by exp4_truth.py with the networks, and its unadjusted correlation
    J = json.load(open(TRUTH.with_suffix(".json"))); r0 = pd.DataFrame({"psi": T.psi, "d": T.d, "rho": T.obs_rho, "se": T.obs_se})
    r0 = pd.concat([r0, pd.DataFrame([{"psi": "none", "d": 0, "rho": J["unadjusted_obs"], "se": 0.0}])], ignore_index=True)
FAM = {"mean_last": ("mean, last layer", P.CAT[0], "-"), "mean_mid": ("mean, middle layer", P.CAT[0], "--"),
       "tok_last": ("last token, last layer", P.CAT[6], "-"), "tok_mid": ("last token, middle layer", P.CAT[6], "--")}
ENC = {"E1": ("$\\psi = E_1$", "s", P.CAT[1]), "E2": ("$\\psi = E_2$", "D", P.CAT[2])}
P.setup()

# ---------------------------------------------------------------- appendix: all four state views, rho_hat and B* against the dimension, and the rejection rates
fig, (ax, bx) = plt.subplots(1, 2, figsize=(P.DOUBLE[0], 2.5), gridspec_kw={"wspace": 0.32})
for name, (lab, col, ls) in FAM.items():
    d = r0[r0.psi.str.startswith(name + ":")].copy()
    if not len(d): continue
    d["dim"] = d.psi.str.split(":").str[1].astype(int); d = d.sort_values("dim")
    ax.errorbar(d.dim, d.rho, yerr=1.96 * d.se, marker="o", ms=3.2, capsize=2, color=col, ls=ls, lw=1.2, label=lab)
    t = T[T.fam == name].copy(); t["dim"] = t.psi.str.split(":").str[1].astype(int); t = t.sort_values("dim")
    ax.plot(t.dim, t.B_star, marker="o", ms=3.4, mfc="white", mew=1.1, color=col, ls=ls, lw=0.9)
for pn, (lab, mk, col) in ENC.items():
    if not (r0.psi == pn).any(): continue
    e = r0[r0.psi == pn].iloc[0]; ax.errorbar([e.d], [e.rho], yerr=[1.96 * e.se], fmt=mk, ms=4.5, color=col, capsize=2, label=lab)
    if pn in set(T.psi): ax.plot([e.d], [T.set_index("psi").loc[pn, "B_star"]], mk, ms=4.5, mfc="white", mew=1.1, color=col)
ax.plot([], [], "o", ms=3.4, mfc="white", mew=1.1, color=P.INK2, label="true insufficiency $\\sqrt{R^2_\\psi(X) R^2_\\psi(Y)}$")
m = float(r0[r0.psi == "none"].rho.iloc[0]); ax.axhline(m, color=P.MUTED, ls=":", lw=0.9)
ax.annotate("unadjusted correlation", (64, m), xytext=(2, 3), textcoords="offset points", fontsize=6.5, color=P.INK2)
ax.axhline(0, color=P.AXIS, lw=0.6); ax.set_xscale("log", base=2); ax.set_xticks([64, 128, 256, 384, 768]); ax.set_xticklabels(["64", "128", "256", "384", "768"]); ax.minorticks_off()
ax.set_xlabel("dimension of $\\psi(Z)$"); ax.set_ylabel("$\\hat\\rho_\\psi$ under $H_0$"); ax.set_ylim(0, m + 0.06)
ax.legend(fontsize=5.6, loc="lower left", ncol=2, frameon=False, handlelength=2.2, columnspacing=0.9)

for name in ("mean_last", "tok_last"):
    t = T[T.fam == name].copy(); t["dim"] = t.psi.str.split(":").str[1].astype(int); t = t.sort_values("dim"); lab, col, _ = FAM[name]
    for key, ls, what in (("rej_std_res", "-", "residual correlation test"), ("rej_Bhalf_res", ":", "tolerance $B^*/2$"), ("rej_lamB_res", "-.", "tolerance $|\\lambda| B^*$"), ("rej_B_res", "--", "tolerance $B^*$")):
        bx.plot(t.dim, t[key], color=col, ls=ls, marker="o", ms=2.8, lw=1.1, label=f"{lab}, {what}")
bx.axhline(0.05, color=P.MUTED, ls=":", lw=0.9); bx.annotate("$\\alpha = 0.05$", (64, 0.05), xytext=(2, 3), textcoords="offset points", fontsize=6.5, color=P.INK2)
bx.set_xscale("log", base=2); bx.set_xticks([64, 128, 256, 384, 768]); bx.set_xticklabels(["64", "128", "256", "384", "768"]); bx.minorticks_off()
bx.set_ylim(-0.03, 1.08); bx.set_xlabel("dimension of $\\psi(Z)$"); bx.set_ylabel("rejection rate under $H_0$")
bx.legend(fontsize=5.0, loc="center left", frameon=False, handlelength=2.4, labelspacing=0.35)
fig.savefig(out("exp4_main"), bbox_inches="tight", pad_inches=0.02); fig.savefig(prev("exp4_main"), dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- appendix: identity check
fig, ax = plt.subplots(figsize=P.SINGLE)
lim = float(max(T.B_star.max(), T.rho_res_mean.abs().max(), T.pred_rho.abs().max()) * 1.12); ax.plot([0, lim], [0, lim], color=P.AXIS, lw=0.8)
for name, (lab, col, ls) in FAM.items():
    t = T[T.fam == name]
    if not len(t): continue
    y, yerr = t.rho_res_mean.abs(), 1.96 * t.rho_res_sd
    ax.errorbar(t.pred_rho.abs(), y, yerr=yerr, fmt="o", color=col, ms=3.6, capsize=1.5, lw=0.8, label=lab)
    ax.errorbar(t.B_star, y, yerr=yerr, fmt="o", mfc="white", mew=1.1, color=col, ms=3.6, capsize=1.5, lw=0.8)
for pn, (lab, mk, col) in ENC.items():
    if pn not in set(T.psi): continue
    t = T.set_index("psi").loc[pn]; y, yerr = abs(t.rho_res_mean), 1.96 * t.rho_res_sd
    ax.errorbar([abs(t.pred_rho)], [y], yerr=[yerr], fmt=mk, color=col, ms=4.5, capsize=1.5, lw=0.8, label=lab)
    ax.errorbar([t.B_star], [y], yerr=[yerr], fmt=mk, mfc="white", mew=1.1, color=col, ms=4.5, capsize=1.5, lw=0.8)
ax.plot([], [], "o", color=P.INK2, ms=3.6, label="prediction $|\\lambda|\\sqrt{R^2_\\psi(X) R^2_\\psi(Y)}$"); ax.plot([], [], "o", mfc="white", mew=1.1, color=P.INK2, ms=3.6, label="bound $\\sqrt{R^2_\\psi(X) R^2_\\psi(Y)}$")
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("identified prediction and bound"); ax.set_ylabel("$\\hat\\rho_\\psi$ under $H_0$")
ax.legend(fontsize=5.6, frameon=False, loc="upper left", handlelength=1.6)
fig.savefig(out("exp4_identity"), bbox_inches="tight", pad_inches=0.02); fig.savefig(prev("exp4_identity"), dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# ---------------------------------------------------------------- main text: level (left) and power (right), sized like the application figure
VIEW = "mean_last"                                   # the state view shown in the main text; the others behave alike (appendix figure)
POWER_VIEW = "E1"                                    # the embedding of the power panel; the others are in the appendix figure
POWER_XMAX = 0.6                                     # right end of the power panel's axis (author, 7 Oct 2026); the appendix figure shows the full range
TESTS = (("rej_std", P.CAT[7], "residual correlation test"), ("rej_Bhalf", P.CAT[3], "tolerance $B^*/2$"),
         ("rej_lamB", P.CAT[0], "tolerance $|\\lambda| B^*$"), ("rej_B", P.CAT[2], "tolerance $B^*$"))
ORACLE = (P.INK2, "--", "post observed (oracle)")
W = pd.read_csv(POWER)
t = T[T.fam == VIEW].copy(); t["dim"] = t.psi.str.split(":").str[1].astype(int); t = t[t.dim <= 384].sort_values("dim")   # 64 to 384 components in the main text
cats = [str(d) for d in t.dim] + [pn for pn in ("E1", "E2") if pn in set(T.psi)]
xs_state = np.arange(len(t)); xpos = {c: i for i, c in enumerate(cats)}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.2, 1.5), sharey=True, gridspec_kw={"wspace": 0.07, "width_ratios": [1, 1.3]})
for key, col, lab in TESTS:
    ax.plot(xs_state, t[key + "_res"], color=col, marker="o", ms=3.0, lw=1.1, label=lab)
    for pn, mk in (("E1", "s"), ("E2", "D")):
        if pn in xpos: ax.plot([xpos[pn]], [T.set_index("psi").loc[pn, key + "_res"]], mk, color=col, ms=4.2)
ax.plot([], [], color=ORACLE[0], ls=ORACLE[1], lw=1.0, label=ORACLE[2])
ax.axhline(0.05, color=P.MUTED, ls=":", lw=0.9); ax.annotate("$\\alpha = 0.05$", (-0.4, 0.05), xytext=(0, 3), textcoords="offset points", fontsize=6, color=P.INK2, ha="left")
ax.axvline(len(t) - 0.5, color=P.AXIS, lw=0.6)
ax.set_xticks(range(len(cats))); ax.set_xticklabels([("$E_1$" if c == "E1" else "$E_2$" if c == "E2" else c) for c in cats], fontsize=6.5)
ax.set_xlabel("embedding $\\psi$", fontsize=7); ax.set_ylabel("rejection rate"); ax.set_ylim(-0.03, 1.08); ax.set_xlim(-0.5, len(cats) - 0.5)
ax.set_title("under $H_0$", fontsize=7, loc="left", pad=3); ax.legend(fontsize=5.5, loc="center", frameon=False, handlelength=1.9, labelspacing=0.2, borderaxespad=0.2)
W = W[W.rho_id >= 0]                                 # the paper shows positive dependence only (the direction of the bias here)
w = W[W.psi == POWER_VIEW].sort_values("rho_id")
bx.plot(w.rho_id, w.rej_oracle, color=ORACLE[0], ls=ORACLE[1], lw=1.0, zorder=5)
for key, col, lab in TESTS: bx.plot(w.rho_id, w[key], color=col, lw=1.2)
bx.axhline(0.05, color=P.MUTED, ls=":", lw=0.9)
bx.set_xlabel("conditional correlation $\\rho_{\\mathrm{id}}$ given the post", fontsize=7); bx.set_xlim(-0.01, POWER_XMAX)   # the main text shows the range where the tests differ
bx.set_title("alternatives, $\\psi = E_1$", fontsize=7, loc="left", pad=3); bx.tick_params(labelleft=False)
fig.savefig(out("exp4_rejection"), bbox_inches="tight", pad_inches=0.02); fig.savefig(prev("exp4_rejection"), dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# ---------------------------------------------------------------- appendix: power for every embedding of exp4_power.py
views = [v for v in ("E1", "E2", "mean_last:384") if v in set(W.psi)]
fig, axs = plt.subplots(1, len(views), figsize=(P.DOUBLE[0], 1.9), sharey=True, gridspec_kw={"wspace": 0.08})
for ax_, v in zip(np.atleast_1d(axs), views):
    w = W[W.psi == v].sort_values("rho_id"); ax_.plot(w.rho_id, w.rej_oracle, color=ORACLE[0], ls=ORACLE[1], lw=1.0, label=ORACLE[2], zorder=5)
    for key, col, lab in TESTS: ax_.plot(w.rho_id, w[key], color=col, lw=1.2, label=lab)
    ax_.axhline(0.05, color=P.MUTED, ls=":", lw=0.9); ax_.set_xlim(-0.01, w.rho_id.max() + 0.02); ax_.set_xlabel("$\\rho_{\\mathrm{id}}$ of the alternative", fontsize=7)
    ax_.set_title({"E1": "$\\psi = E_1$", "E2": "$\\psi = E_2$", "mean_last:384": "$\\psi$ = 384 components of the mean state, last layer"}[v], fontsize=7, loc="left", pad=3)
np.atleast_1d(axs)[0].set_ylabel("rejection rate"); np.atleast_1d(axs)[0].set_ylim(-0.03, 1.08)
h, l = np.atleast_1d(axs)[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=6, frameon=False, handlelength=1.9, columnspacing=1.2)
fig.savefig(out("exp4_power_views"), bbox_inches="tight", pad_inches=0.02); fig.savefig(prev("exp4_power_views"), dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print(f"written ({a.regressor}): {out('exp4_rejection').name} (main text: level and power), {out('exp4_main').name}, {out('exp4_power_views').name}, {out('exp4_identity').name} (appendix) in {PAPER / 'figures'}")
