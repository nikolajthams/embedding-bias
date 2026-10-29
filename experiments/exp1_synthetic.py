"""Experiment 1: the linear-Gaussian latent model.

Phases (each writes per-replication raw outputs to code/results/exp1_*_raw.csv):
  typeI  -> exp1_typeI_raw.csv           (standard test, four configurations, n grid)
  bound  -> exp1_bound_tightness.csv     (population |rho| vs bound, MC check at n = 20000)
  robust -> exp1_robust_test_raw.csv     (lam = 1, with the psi_5 -> psi_10 benchmark)
  power  -> exp1_power_raw.csv           (sufficient psi, n = 1000, effect grid)
  plots  -> all figures and aggregate CSVs, recomputed from the raw files with the current
            definitions of the robust test (exact least-favourable calibration) and RV_alpha.

usage: python experiments/exp1_synthetic.py [N_REP] [--only typeI,bound,robust,power,plots]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from citext import plotting as P  # noqa: E402
from citext.estimators import benchmark_pair, gcm_test, robust_test, rv_alpha  # noqa: E402
from citext.paths import RESULTS_DIR  # noqa: E402
from citext.synthetic import LinearConfig, population, partial_r2_population, sample  # noqa: E402

ALPHA = 0.05
NS = [500, 1000, 2000, 4000, 10000]   # the sample sizes of Experiment 2 (author, 30 Sept 2026)
N_JOBS = 8
REG = "ols"   # least squares throughout (at most 20 columns, at least 400 training rows)
R2 = 0.1
Q_BENCH = 10  # benchmark encoder psi' = psi_10 (coordinates 0..9) for psi = psi_5

CONFIGS_TYPEI = {
    "sufficient (q=20)": LinearConfig(q=20),
    "q=5, lam=0": LinearConfig(q=5, r2_A=R2, r2_B=R2, lam=0.0),
    "q=5, lam=0.5": LinearConfig(q=5, r2_A=R2, r2_B=R2, lam=0.5),
    "q=5, lam=1": LinearConfig(q=5, r2_A=R2, r2_B=R2, lam=1.0),
}
BETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]


def cfg_record(cfg: LinearConfig) -> dict:
    pop = population(cfg)
    rec = {"q": cfg.q, "lam": cfg.lam, "beta": cfg.beta, "r2_A_pop": pop["r2_A"], "r2_B_pop": pop["r2_B"],
           "rho_pop": pop["rho"], "bound_pop": pop["bound"]}
    if cfg.q < Q_BENCH:
        rec["bench_pop"] = partial_r2_population(cfg, Q_BENCH)["bench"]
    return rec


def one_rep(cfg: LinearConfig, n: int, seed: int, with_bench: bool = False):
    rng = np.random.default_rng(seed)
    d = sample(cfg, n, rng)
    res = gcm_test(d["A"], d["B"], d["Psi"], regressor=REG, seed=seed, alpha=ALPHA)
    out = {"n": n, "seed": seed, "rho_hat": res.rho_hat, "se_hat": res.se, "pvalue": res.pvalue}
    if with_bench:
        b = benchmark_pair(d["A"], d["B"], d["Psi"], d["Z"][:, cfg.q:Q_BENCH], regressor=REG, seed=seed)
        out.update({"bench_r2_A_hat": b["r2_A"], "bench_r2_B_hat": b["r2_B"], "bench_hat": b["bench"]})
    return out


def run_grid(cfg, ns, n_rep, with_bench=False, tag=""):
    rows = []
    rec = cfg_record(cfg)
    for n in ns:
        t = time.time()
        res = Parallel(n_jobs=N_JOBS)(delayed(one_rep)(cfg, n, s, with_bench) for s in range(n_rep))
        for r in res:
            rows.append({"tag": tag, **rec, **r})
        print(f"  {tag} n={n}: {time.time()-t:.1f}s", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- phases
def phase_typeI(n_rep):
    print("phase typeI", flush=True)
    df = pd.concat([run_grid(cfg, NS, n_rep, tag=tag) for tag, cfg in CONFIGS_TYPEI.items()])
    df.to_csv(RESULTS_DIR / "exp1_typeI_raw.csv", index=False)


def phase_bound():
    print("phase bound", flush=True)
    rows = []
    rng = np.random.default_rng(777)
    for r2a in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for r2b in [r2a, min(0.5, r2a * 2)]:
            for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
                cfg = LinearConfig(q=5, r2_A=r2a, r2_B=r2b, lam=lam)
                pop = population(cfg)
                d = sample(cfg, 20000, rng)
                res = gcm_test(d["A"], d["B"], d["Psi"], regressor=REG, seed=0)
                rows.append({"r2_A": r2a, "r2_B": r2b, "lam": lam, "bound": pop["bound"], "rho_pop": pop["rho"],
                             "rho_hat_n20000": res.rho_hat, "se_hat_n20000": res.se})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp1_bound_tightness.csv", index=False)


def phase_robust(n_rep):
    print("phase robust", flush=True)
    cfg = LinearConfig(q=5, r2_A=R2, r2_B=R2, lam=1.0)
    df = run_grid(cfg, NS, n_rep, with_bench=True, tag="q=5, lam=1")
    df.to_csv(RESULTS_DIR / "exp1_robust_test_raw.csv", index=False)


def phase_power(n_rep):
    print("phase power", flush=True)
    frames = []
    for beta in BETAS:
        cfg = LinearConfig(q=20, beta=beta)
        frames.append(run_grid(cfg, [1000], n_rep, tag=f"beta={beta}"))
    pd.concat(frames).to_csv(RESULTS_DIR / "exp1_power_raw.csv", index=False)


# --------------------------------------------------------------------------- plots
def log_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel(r"sample size $n$")


def naive_reject(df):
    from scipy import stats
    return (np.abs(df.rho_hat) > stats.norm.ppf(1 - ALPHA / 2) * df.se_hat).astype(float)


def robust_reject(df, tol):
    r0 = np.broadcast_to(np.asarray(tol, dtype=float), (len(df),))
    return np.array([robust_test(r, s, x, ALPHA)["reject"] for r, s, x in zip(df.rho_hat, df.se_hat, r0)], dtype=float)


def phase_plots():
    import matplotlib.pyplot as plt
    P.setup()
    summary = {"regressor": REG, "alpha": ALPHA, "ns": NS, "r2": R2, "p": 20, "q_insufficient": 5, "q_sufficient": 20,
               "bench_encoder": f"psi_{Q_BENCH} (first {Q_BENCH} coordinates) as benchmark psi' for psi_5",
               "effect_grid_gamma": BETAS, "power_n": 1000, "bound_mc_n": 20000, "K_folds": 5}

    # ---- Figure 1: type I error vs n
    df = pd.read_csv(RESULTS_DIR / "exp1_typeI_raw.csv")
    df["reject"] = naive_reject(df)
    n_rep = int(df.groupby(["tag", "n"]).size().min())
    summary["n_rep"] = n_rep
    summary["n_rep_per_setting"] = {"typeI": n_rep, "robust": int(pd.read_csv(RESULTS_DIR / "exp1_robust_test_raw.csv").groupby("n").size().min()),
                                    "power": int(pd.read_csv(RESULTS_DIR / "exp1_power_raw.csv").groupby("beta").size().min())}
    agg = df.groupby(["tag", "n"]).agg(rej=("reject", "mean"), rho_pop=("rho_pop", "first"),
                                       rho_hat_mean=("rho_hat", "mean")).reset_index()
    agg["mc_se"] = np.sqrt(agg.rej * (1 - agg.rej) / n_rep)
    agg.to_csv(RESULTS_DIR / "exp1_typeI_vs_n.csv", index=False)
    fig, ax = plt.subplots(figsize=P.SINGLE)
    seq = P.seq_colors(3)
    style = {
        "sufficient (q=20)": dict(color=P.CAT[2], marker="s", linestyle="--", label=r"sufficient $\psi$", zorder=4),
        "q=5, lam=0": dict(color=seq[0], marker="o", linestyle=":", markerfacecolor="white", markeredgecolor=seq[0],
                           markeredgewidth=1.2, label=r"insufficient, $\lambda=0$", zorder=5),
        "q=5, lam=0.5": dict(color=seq[1], marker="o", label=r"insufficient, $\lambda=0.5$"),
        "q=5, lam=1": dict(color=seq[2], marker="o", label=r"insufficient, $\lambda=1$"),
    }
    for tag, st in style.items():
        a = agg[agg.tag == tag].sort_values("n")
        ax.plot(a.n, a.rej, **st)
    ax.axhline(ALPHA, color=P.MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate(r"$\alpha=0.05$", (NS[-1], ALPHA), xytext=(-2, 3), textcoords="offset points", fontsize=6.5, color=P.INK2, ha="right")
    log_axis(ax)
    ax.set_ylabel("rejection rate under $H_0$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    P.savefig(fig, "exp1_typeI_vs_n.pdf")
    summary["fig1"] = {f"{t}|n={int(n)}": float(r) for t, n, r in zip(agg.tag, agg.n, agg.rej)}

    # ---- Figure 2: bound tightness
    df2 = pd.read_csv(RESULTS_DIR / "exp1_bound_tightness.csv")
    lam_grid = sorted(df2.lam.unique())
    fig, ax = plt.subplots(figsize=P.SINGLE)
    seq5 = P.seq_colors(5)
    xx = np.linspace(0, 0.52, 10)
    ax.plot(xx, xx, color=P.MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate(r"bound $\sqrt{R^2_\psi(X)\,R^2_\psi(Y)}$", (0.45, 0.45), xytext=(-4, 5), textcoords="offset points", fontsize=6.5,
                color=P.INK2, rotation=45, rotation_mode="anchor", ha="right", va="bottom")
    for c, lam in zip(seq5, lam_grid):
        a = df2[df2.lam == lam].sort_values("bound")
        ax.plot(a.bound, np.abs(a.rho_pop), color=c, linewidth=1.2, zorder=2)
        ax.plot(a.bound, np.abs(a.rho_hat_n20000), linestyle="none", marker="o", color=c, markersize=4.5, zorder=3,
                label=rf"$\lambda={lam}$")
    ax.set_xlabel(r"$\sqrt{R^2_\psi(X)\,R^2_\psi(Y)}$")
    ax.set_ylabel(r"$|\rho_\psi|$ under $H_0$")
    ax.set_xlim(0, 0.53)
    ax.set_ylim(-0.01, 0.53)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", title="alignment", title_fontsize=7)
    P.savefig(fig, "exp1_bound_tightness.pdf")
    summary["fig2"] = {"max_abs_dev_mc_vs_pop": float(np.max(np.abs(df2.rho_hat_n20000 - df2.rho_pop)))}

    # ---- Figure 3 and 5: robust test and RV distribution (lam = 1)
    df3 = pd.read_csv(RESULTS_DIR / "exp1_robust_test_raw.csv")
    bound, bench_pop = float(df3.bound_pop.iloc[0]), float(df3.bench_pop.iloc[0])
    df3["naive"] = naive_reject(df3)
    df3["robust_bound"] = robust_reject(df3, bound)
    df3["robust_half"] = robust_reject(df3, bound / 2)
    df3["robust_bench"] = robust_reject(df3, df3.bench_hat.values)
    df3["RV"] = np.abs(df3.rho_hat)
    df3["RV_alpha"] = [rv_alpha(r, s, ALPHA) for r, s in zip(df3.rho_hat, df3.se_hat)]
    df3.to_csv(RESULTS_DIR / "exp1_robust_test_decisions.csv", index=False)
    agg3 = df3.groupby("n").agg(naive=("naive", "mean"), robust_bound=("robust_bound", "mean"), robust_half=("robust_half", "mean"),
                                robust_bench=("robust_bench", "mean"), bench_median=("bench_hat", "median"),
                                RV_median=("RV", "median"), RV_alpha_median=("RV_alpha", "median")).reset_index()
    agg3.to_csv(RESULTS_DIR / "exp1_robust_test.csv", index=False)
    fig, ax = plt.subplots(figsize=P.SINGLE)
    for key, lab, c in [("naive", "standard test", P.CAT[1]),
                        ("robust_half", r"robust, $B=$ half the bound", P.CAT[3]),
                        ("robust_bound", r"robust, $B=$ bound", P.CAT[0])]:
        ax.plot(agg3.n, agg3[key], marker="o", color=c, label=lab)
    ax.axhline(ALPHA, color=P.MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate(r"$\alpha=0.05$", (NS[0], ALPHA), xytext=(2, 3), textcoords="offset points", fontsize=6.5, color=P.INK2)
    log_axis(ax)
    ax.set_ylabel("rejection rate under $H_0$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    P.savefig(fig, "exp1_robust_test.pdf")
    summary["fig3"] = {"bound": bound, "bench_pop": bench_pop, "rows": agg3.to_dict(orient="records")}

    fig, ax = plt.subplots(figsize=P.SINGLE)
    pos, w = np.arange(len(NS)), 0.36
    kw = dict(showfliers=False, patch_artist=True, medianprops={"color": P.INK, "linewidth": 0.9})
    bp1 = ax.boxplot([df3[df3.n == n].RV.values for n in NS], positions=pos - w / 2, widths=w * 0.9,
                     whiskerprops={"color": P.CAT[0], "linewidth": 0.8}, capprops={"color": P.CAT[0], "linewidth": 0.8},
                     boxprops={"facecolor": P.CAT[0], "edgecolor": P.CAT[0], "linewidth": 0.8, "alpha": 0.85}, **kw)
    bp2 = ax.boxplot([df3[df3.n == n].RV_alpha.values for n in NS], positions=pos + w / 2, widths=w * 0.9,
                     whiskerprops={"color": P.SEQ[250], "linewidth": 0.8}, capprops={"color": P.SEQ[250], "linewidth": 0.8},
                     boxprops={"facecolor": P.SEQ[250], "edgecolor": P.SEQ[250], "linewidth": 0.8}, **kw)
    ax.axhline(abs(df3.rho_pop.iloc[0]), color=P.MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate(r"$|\rho_\psi|$", (pos[-1] + 0.45, abs(df3.rho_pop.iloc[0])), xytext=(2, 2), textcoords="offset points",
                fontsize=6.5, color=P.INK2)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel(r"sample size $n$")
    ax.set_ylabel("robustness value")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], [r"$\mathrm{RV}=|\hat\rho|$", r"$\mathrm{RV}_\alpha$"], loc="upper right")
    ax.grid(axis="x", visible=False)
    P.savefig(fig, "exp1_rv_estimation.pdf")
    summary["fig5"] = {f"n={int(n)}": {"RV_median": float(df3[df3.n == n].RV.median()),
                                       "RV_alpha_median": float(df3[df3.n == n].RV_alpha.median())} for n in NS}

    # ---- Figure 4: power
    df4 = pd.read_csv(RESULTS_DIR / "exp1_power_raw.csv")
    df4["naive"] = naive_reject(df4)
    for r0 in (0.05, 0.1, 0.2):
        df4[f"robust_{r0}"] = robust_reject(df4, r0)
    agg4 = df4.groupby("beta").agg(rho_pop=("rho_pop", "first"), naive=("naive", "mean"), **{f"robust_{r0}": (f"robust_{r0}", "mean") for r0 in (0.05, 0.1, 0.2)}).reset_index()
    agg4.to_csv(RESULTS_DIR / "exp1_power.csv", index=False)
    fig, ax = plt.subplots(figsize=P.SINGLE)
    ax.plot(agg4.beta, agg4.naive, marker="o", color=P.CAT[1], label="standard test")
    for c, r0 in zip(P.seq_colors(3), (0.05, 0.1, 0.2)):
        ax.plot(agg4.beta, agg4[f"robust_{r0}"], marker="o", color=c, label=rf"robust, $B={r0}$")
    ax.axhline(ALPHA, color=P.MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel(r"effect size $\gamma$ ($Y \leftarrow Y + \gamma X$)")
    ax.set_ylabel("rejection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center right")
    P.savefig(fig, "exp1_power.pdf")
    summary["fig4"] = agg4.to_dict(orient="records")
    (RESULTS_DIR / "summary_exp1.json").write_text(json.dumps(summary, indent=1, default=float))
    print("plots written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_rep", nargs="?", type=int, default=500)
    ap.add_argument("--only", default="typeI,bound,robust,power,plots")
    args = ap.parse_args()
    phases = args.only.split(",")
    t0 = time.time()
    if "typeI" in phases:
        phase_typeI(args.n_rep)
    if "bound" in phases:
        phase_bound()
    if "robust" in phases:
        phase_robust(args.n_rep)
    if "power" in phases:
        phase_power(args.n_rep)
    if "plots" in phases:
        phase_plots()
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
