"""Experiment 2: real text (20 Newsgroups) as Z with synthetic outcomes.

Phases (raw per-replication outputs in code/results/exp2_*_raw.csv; figures recomputed from raw
in the 'plots' phase with the current robust-test calibration and RV_alpha):
  pop    -> corpus-level quantities for psi = E1 (cached under ~/.cache/ci_text)
  typeI  -> exp2_text_typeI_raw.csv
  bench  -> exp2_text_rv_benchmark_raw.csv   (psi = E1, benchmark psi' = E2, lam = 1)
  power  -> exp2_text_power_raw.csv           (psi = E1, n = POWER_N, lam in {0, 1}, effect grid)
  plots  -> exp2_text_typeI.pdf, exp2_text_rv_benchmark.pdf, exp2_text_power.pdf
Regressors. Every raw file holds, per draw, least squares ('ols'), RidgeCV ('ridge') and kernel ridge (RBF, median
heuristic, 'krr', for n <= KRR_MAX_N only), all on the same folds. The paper's primary regressor is 'hybrid': least
squares when every training fold has at least HYBRID_MIN_ROWS_PER_COL (= 1) rows per column of the conditioning set,
i.e. n >= d on the grid, and RidgeCV otherwise (citext.estimators, 7 Oct 2026; from 30 Sept to 7 Oct least squares only
with smaller configurations omitted, before that the same rule with threshold 2.5). The hybrid columns are derived from
the stored columns in the plots phase, so the threshold can be changed without re-running; the two regressions of a
benchmark are resolved together by the richer set.
Design without the TF-IDF block since 29 Sept 2026 (build_design(with_S=False)).

usage: python experiments/exp2_text.py [N_REP] [--only pop,typeI,bench,power,plots]
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
from scipy import stats
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from citext import plotting as P  # noqa: E402
from citext.estimators import HYBRID_MIN_ROWS_PER_COL, benchmark_pair, gcm_test, hybrid_choice_n, robust_test, rv_alpha  # noqa: E402
from citext.paths import RESULTS_DIR  # noqa: E402
from citext.text_semi import (E1_NAME, E2_NAME, build_design, oracle_encoder, outcomes,  # noqa: E402
                              population_quantities)

ALPHA = 0.05
NS = [500, 1000, 2000, 4000, 10000]
R2 = 0.1
N_JOBS = 5
BLAS_THREADS = 2
PRIMARY, SECONDARY = "hybrid", "krr"   # primary: least squares or ridge by the rule in citext.estimators (derived at plot time); secondary: kernel ridge
RAW_REGRESSORS = ("ols", "ridge", "krr")   # stored per draw in every raw file, same folds
ALL_REGRESSORS = tuple(dict.fromkeys((PRIMARY,) + RAW_REGRESSORS))
KRR_MAX_N = 4000                       # kernel ridge is skipped (NaN) above this n
SKIP_KRR = __import__("os").environ.get("EXP2_SKIP_KRR", "0") == "1"   # skip the kernel ridge secondary in a compute phase (power at n = 2000: it takes 13 min per configuration)
K_FOLDS = 5
RICH_MAX_REP = 200                     # draws for the high-dimensional set in the type-I phase (kernel ridge makes it slow)
GAMMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
POWER_N = 2000   # 1000 until 30 Sept 2026, when it was moved to the smallest n reported under the rule of that date

DESIGN = build_design(0, with_S=False)
N_ALL = len(DESIGN["g1"])


def psi_matrix(name, idx):
    if name == "E1":
        return DESIGN["E1"][idx]
    if name == "E2":
        return DESIGN["E2"][idx]
    if name == "oracle":
        return oracle_encoder(DESIGN, idx)
    if name == "rich":
        return np.hstack([DESIGN["E1"][idx], DESIGN["E2"][idx]])
    raise ValueError(name)


def rep_typeI(psi_name, lam, n, seed, gamma=0.0):
    with threadpool_limits(BLAS_THREADS):
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_ALL, n, replace=False)
        A, B = outcomes(DESIGN, idx, lam, R2, R2, gamma, rng)
        Psi = psi_matrix(psi_name, idx)
        out = {"psi": psi_name, "lam": lam, "n": n, "seed": seed, "gamma": gamma, "d": int(Psi.shape[1])}
        for reg in RAW_REGRESSORS:
            if reg == "krr" and (n > KRR_MAX_N or SKIP_KRR):
                out.update({f"rho_{reg}": np.nan, f"se_{reg}": np.nan})
                continue
            r = gcm_test(A, B, Psi, regressor=reg, seed=seed, alpha=ALPHA)
            out.update({f"rho_{reg}": r.rho_hat, f"se_{reg}": r.se})
        return out


def rep_bench(lam, n, seed, gamma=0.0):
    """psi = E1, benchmark encoder psi' = E2."""
    with threadpool_limits(BLAS_THREADS):
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_ALL, n, replace=False)
        A, B = outcomes(DESIGN, idx, lam, R2, R2, gamma, rng)
        out = {"psi": "E1", "lam": lam, "n": n, "seed": seed, "gamma": gamma,
               "d0": int(DESIGN["E1"].shape[1]), "d1": int(DESIGN["E1"].shape[1] + DESIGN["E2"].shape[1])}
        for reg in RAW_REGRESSORS:
            keys = [f"{k}_{reg}" for k in ("rho", "se", "bench", "benchA", "benchB", "mse0A", "mse1A", "mse0B", "mse1B")]
            if reg == "krr" and (n > KRR_MAX_N or SKIP_KRR):
                out.update({k: np.nan for k in keys})
                continue
            b = benchmark_pair(A, B, DESIGN["E1"][idx], DESIGN["E2"][idx], regressor=reg, seed=seed)
            g = b["gcm"]
            out.update(dict(zip(keys, [g.rho_hat, g.se, b["bench"], b["r2_A"], b["r2_B"], b["mse0_A"], b["mse1_A"], b["mse0_B"], b["mse1_B"]])))
        return out


def get_pop():
    pop = {}
    for lam in (0.0, 1.0):
        for reg in ALL_REGRESSORS:
            pop[f"lam{lam}_{reg}"] = population_quantities(DESIGN, lam, R2, R2, regressor=reg, n_max=8000)
    return pop


# --------------------------------------------------------------------------- phases
def phase_typeI(n_rep):
    print("phase typeI", flush=True)
    configs = [("E1", 1.0, n_rep), ("E1", 0.0, n_rep), ("oracle", 1.0, n_rep), ("rich", 1.0, min(n_rep, RICH_MAX_REP))]
    pop = get_pop()
    rows = []
    for psi, lam, nrep in configs:
        for n in NS:
            t = time.time()
            res = Parallel(n_jobs=N_JOBS)(delayed(rep_typeI)(psi, lam, n, s) for s in range(nrep))
            for r in res:
                pk = pop[f"lam{lam}_{PRIMARY}"]
                r.update({f"rho_pop_E1_{PRIMARY}": pk["rho"], f"bound_pop_E1_{PRIMARY}": pk["bound"]})
            rows += res
            print(f"  {psi} lam={lam} n={n}: {time.time()-t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp2_text_typeI_raw.csv", index=False)


def phase_bench(n_rep):
    print("phase bench", flush=True)
    pop = get_pop()[f"lam1.0_{PRIMARY}"]
    rows = []
    for n in NS:
        t = time.time()
        res = Parallel(n_jobs=N_JOBS)(delayed(rep_bench)(1.0, n, s) for s in range(n_rep))
        for r in res:
            r.update({"rho_pop": pop["rho"], "bound_pop": pop["bound"], "bench_pop": pop["bench"]})
        rows += res
        print(f"  bench n={n}: {time.time()-t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp2_text_rv_benchmark_raw.csv", index=False)


def phase_power(n_rep):
    print("phase power", flush=True)
    pop = get_pop()
    rows = []
    for lam in (0.0, 1.0):
        for gamma in GAMMAS:
            t = time.time()
            res = Parallel(n_jobs=N_JOBS)(delayed(rep_bench)(lam, POWER_N, s, gamma) for s in range(n_rep))
            pk = pop[f"lam{lam}_{PRIMARY}"]
            for r in res:
                r.update({"rho_pop_H0": pk["rho"], "bound_pop": pk["bound"], "bench_pop": pk["bench"]})
            rows += res
            print(f"  power lam={lam} gamma={gamma}: {time.time()-t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp2_text_power_raw.csv", index=False)


# --------------------------------------------------------------------------- plots
def naive_reject(rho, se):
    rho, se = np.asarray(rho, dtype=float), np.asarray(se, dtype=float)
    out = (np.abs(rho) > stats.norm.ppf(1 - ALPHA / 2) * se).astype(float)
    out[~np.isfinite(rho)] = np.nan
    return out


def robust_reject(rho, se, tol):
    r0 = np.broadcast_to(np.asarray(tol, dtype=float), (len(rho),))
    return np.array([robust_test(r, s, x, ALPHA)["reject"] if np.isfinite(r) and np.isfinite(s) and np.isfinite(x) else np.nan
                     for r, s, x in zip(rho, se, r0)], dtype=float)


def choice(n, d):
    """The regressor of a configuration with n rows on a d-dimensional conditioning set under the rule (smallest training fold)."""
    return hybrid_choice_n(int(n), int(d), K_FOLDS)


def add_hybrid_typeI(df):
    """Derive the 'hybrid' columns of a type-I raw table from the stored 'ols' and 'ridge' columns by the rule."""
    ols = np.array([choice(n, d) == "ols" for n, d in zip(df.n, df.d)])
    for col in ("rho", "se"):
        df[f"{col}_hybrid"] = np.where(ols, df[f"{col}_ols"], df[f"{col}_ridge"])
    df["hybrid_choice"] = np.where(ols, "ols", "ridge")
    return df


def add_hybrid_bench(df):
    """Derive the 'hybrid' columns of a benchmark/power raw table. The test statistic (rho, se) comes from the base
    regression on E1 (d0 columns) resolved by the rule for d0, as in the type-I phase. The benchmark compares the
    out-of-fold MSEs of the base and the richer regression, which must use the same procedure, so both are resolved
    by the rule for the richer set (d1 columns); mixing least squares and ridge would not give a lower bound."""
    ols0 = np.array([choice(n, d) == "ols" for n, d in zip(df.n, df.d0)])
    ols1 = np.array([choice(n, d) == "ols" for n, d in zip(df.n, df.d1)])
    for col in ("rho", "se"):
        df[f"{col}_hybrid"] = np.where(ols0, df[f"{col}_ols"], df[f"{col}_ridge"])
    for y in ("A", "B"):
        mse0 = np.where(ols1, df[f"mse0{y}_ols"], df[f"mse0{y}_ridge"])
        mse1 = np.where(ols1, df[f"mse1{y}_ols"], df[f"mse1{y}_ridge"])
        df[f"bench{y}_hybrid"] = np.maximum(0.0, (mse0 - mse1) / mse0)
    df["bench_hybrid"] = np.sqrt(df["benchA_hybrid"] * df["benchB_hybrid"])
    df["hybrid_choice_test"] = np.where(ols0, "ols", "ridge")
    df["hybrid_choice_bench"] = np.where(ols1, "ols", "ridge")
    return df


def log_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel(r"sample size $n$")


def phase_plots():
    import matplotlib.pyplot as plt
    P.setup()
    pop = get_pop()
    summary = {"primary_regressor": PRIMARY, "secondary_regressor": SECONDARY, "r2": R2, "alpha": ALPHA, "ns": NS,
               "n_corpus_posts": int(N_ALL), "corpus": "20 Newsgroups (headers/footers/quotes removed, >= 30 words, truncated to 300 words)",
               "encoders": {"E1": E1_NAME, "E2": E2_NAME}, "dim_E1": int(DESIGN["E1"].shape[1]), "dim_E2": int(DESIGN["E2"].shape[1]),
               "design": "g1, g2 = standardized random projections of the residual of E2 after least squares on E1 over the corpus (no TF-IDF block)",
               "rich_encoder_dim": int(DESIGN["E1"].shape[1] + DESIGN["E2"].shape[1]), "raw_regressors": list(RAW_REGRESSORS), "krr_max_n": KRR_MAX_N,
               "regressor_rule": f"hybrid = least squares when every training fold has at least {HYBRID_MIN_ROWS_PER_COL:g} rows per column (n >= d on the grid), RidgeCV otherwise",
               "hybrid_min_rows_per_col": HYBRID_MIN_ROWS_PER_COL,
               "oracle_encoder": "[h_A, h_B, g1, g2] (4 generating features)", "effect_grid_gamma": GAMMAS, "power_n": POWER_N,
               "population": pop, "population_n_max": 8000, "K_folds": 5,
               "figure_regressors": {"exp2_text_typeI.pdf": PRIMARY, "exp2_text_rv_benchmark.pdf": PRIMARY, "exp2_text_power.pdf": PRIMARY}}

    # ---- type I error
    df = add_hybrid_typeI(pd.read_csv(RESULTS_DIR / "exp2_text_typeI_raw.csv"))
    for reg in ALL_REGRESSORS:
        df[f"reject_{reg}"] = naive_reject(df[f"rho_{reg}"], df[f"se_{reg}"])
    df.to_csv(RESULTS_DIR / "exp2_text_typeI_decisions.csv", index=False)
    agg = df.groupby(["psi", "lam", "n"]).agg(**{f"rej_{reg}": (f"reject_{reg}", "mean") for reg in ALL_REGRESSORS},
                                              **{f"rho_{reg}": (f"rho_{reg}", "mean") for reg in ALL_REGRESSORS},
                                              nrep=("seed", "count"), reg_used=("hybrid_choice", "first")).reset_index()
    agg.to_csv(RESULTS_DIR / "exp2_text_typeI.csv", index=False)
    summary["typeI"] = agg.to_dict(orient="records")
    summary["n_rep_per_setting"] = {"typeI": {f"{p}|lam={l}": int(v) for (p, l), v in df.groupby(["psi", "lam"]).size().div(len(NS)).items()}}
    fig, ax = plt.subplots(figsize=P.SINGLE)
    series = [(("E1", 1.0), r"$\psi=E_1$ (insufficient), $\lambda=1$", P.seq_colors(2)[1], dict(marker="o")),
              (("E1", 0.0), r"$\psi=E_1$ (insufficient), $\lambda=0$", P.seq_colors(2)[0], dict(marker="o", linestyle=":", markerfacecolor="white", markeredgewidth=1.2)),
              (("rich", 1.0), r"$\psi=[E_1,E_2]$ (sufficient, %d-dim)" % summary["rich_encoder_dim"], P.CAT[1], dict(marker="o")),
              (("oracle", 1.0), r"$\psi=$ generating features (sufficient)", P.CAT[2], dict(marker="s", linestyle="--"))]
    for (psi, lam), lab, c, st in series:
        a = agg[(agg.psi == psi) & (agg.lam == lam)].sort_values("n")
        kw = dict(color=c, label=lab, **st)
        if "markerfacecolor" in st:
            kw["markeredgecolor"] = c
        ax.plot(a.n, a[f"rej_{PRIMARY}"], **kw)
    ax.axhline(ALPHA, color=P.MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate(r"$\alpha=0.05$", (NS[-1], ALPHA), xytext=(-2, 3), textcoords="offset points", fontsize=6.5, color=P.INK2, ha="right")
    log_axis(ax)
    ax.set_ylabel("rejection rate under $H_0$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center right", fontsize=6.2, bbox_to_anchor=(1.0, 0.45))
    P.savefig(fig, "exp2_text_typeI.pdf")

    # ---- RV vs benchmark (lam = 1)
    dfB = add_hybrid_bench(pd.read_csv(RESULTS_DIR / "exp2_text_rv_benchmark_raw.csv"))
    for reg in (PRIMARY, SECONDARY):
        dfB[f"RV_{reg}"] = np.abs(dfB[f"rho_{reg}"])
        dfB[f"RValpha_{reg}"] = [rv_alpha(r, s, ALPHA) if np.isfinite(r) and np.isfinite(s) else np.nan
                                 for r, s in zip(dfB[f"rho_{reg}"], dfB[f"se_{reg}"])]
        dfB[f"reject_{reg}"] = naive_reject(dfB[f"rho_{reg}"], dfB[f"se_{reg}"])
        dfB[f"robust_bench_{reg}"] = robust_reject(dfB[f"rho_{reg}"], dfB[f"se_{reg}"], dfB[f"bench_{reg}"].values)
        dfB[f"robust_bound_{reg}"] = robust_reject(dfB[f"rho_{reg}"], dfB[f"se_{reg}"], pop[f"lam1.0_{reg}"]["bound"])
        dfB[f"robust_half_{reg}"] = robust_reject(dfB[f"rho_{reg}"], dfB[f"se_{reg}"], pop[f"lam1.0_{reg}"]["bound"] / 2)
    dfB.to_csv(RESULTS_DIR / "exp2_text_rv_benchmark_decisions.csv", index=False)
    q25, q75 = (lambda s: s.quantile(.25)), (lambda s: s.quantile(.75))
    P1, S1 = PRIMARY, SECONDARY
    aggB = dfB.groupby("n").agg(**{"RV_med": (f"RV_{P1}", "median"), "RV_lo": (f"RV_{P1}", q25), "RV_hi": (f"RV_{P1}", q75),
                                   "RVa_med": (f"RValpha_{P1}", "median"), "bench_med": (f"bench_{P1}", "median"),
                                   "bench_lo": (f"bench_{P1}", q25), "bench_hi": (f"bench_{P1}", q75),
                                   f"RV_med_{S1}": (f"RV_{S1}", "median"), f"bench_med_{S1}": (f"bench_{S1}", "median"),
                                   "naive_rej": (f"reject_{P1}", "mean"), "robust_bench_rej": (f"robust_bench_{P1}", "mean"),
                                   "robust_bound_rej": (f"robust_bound_{P1}", "mean"), "robust_half_rej": (f"robust_half_{P1}", "mean"),
                                   "reg_used_test": ("hybrid_choice_test", "first"), "reg_used_bench": ("hybrid_choice_bench", "first")}).reset_index()
    aggB["frac_RV_above_bench"] = [np.mean(dfB[dfB.n == n][f"RV_{P1}"] > dfB[dfB.n == n][f"bench_{P1}"]) for n in aggB.n]
    aggB["frac_RValpha_above_bench"] = [np.mean(dfB[dfB.n == n][f"RValpha_{P1}"] > dfB[dfB.n == n][f"bench_{P1}"]) for n in aggB.n]
    aggB.to_csv(RESULTS_DIR / "exp2_text_rv_benchmark.csv", index=False)
    summary["rv_benchmark"] = aggB.to_dict(orient="records")
    summary["n_rep_per_setting"]["bench"] = int(dfB.groupby("n").size().min())
    popk = pop[f"lam1.0_{PRIMARY}"]
    fig, ax = plt.subplots(figsize=P.SINGLE)
    ax.fill_between(aggB.n, aggB.RV_lo, aggB.RV_hi, color=P.CAT[0], alpha=0.15, linewidth=0)
    ax.plot(aggB.n, aggB.RV_med, marker="o", color=P.CAT[0], label=r"$\mathrm{RV}=|\hat\rho|$ (median, IQR)")
    ax.plot(aggB.n, aggB.RVa_med, marker="o", color=P.SEQ[250], label=r"$\mathrm{RV}_\alpha$ (median)")
    ax.fill_between(aggB.n, aggB.bench_lo, aggB.bench_hi, color=P.CAT[1], alpha=0.15, linewidth=0)
    ax.plot(aggB.n, aggB.bench_med, marker="s", color=P.CAT[1], label=r"estimated benchmark $\hat\rho_{\mathrm{bench}}$")
    box = dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85)
    ax.axhline(popk["bound"], color=P.INK2, linewidth=0.9, linestyle=(0, (5, 2)), zorder=2)
    if abs(popk["bound"] - popk["bench"]) < 0.003:   # E2 demonstrates all of the insufficiency: one line, one label
        ax.annotate(r"oracle bound $\sqrt{R^2_\psi(X)\,R^2_\psi(Y)}$ $=$ population benchmark", (NS[0], popk["bound"]), xytext=(2, -3),
                    textcoords="offset points", fontsize=6.5, color=P.INK2, ha="left", va="top", bbox=box, zorder=4)
    else:
        ax.annotate(r"oracle bound $\sqrt{R^2_\psi(X)\,R^2_\psi(Y)}$", (NS[0], popk["bound"]), xytext=(2, -3), textcoords="offset points",
                    fontsize=6.5, color=P.INK2, ha="left", va="top", bbox=box, zorder=4)
        ax.axhline(popk["bench"], color=P.CAT[6], linewidth=1.0, linestyle=(0, (1, 1.6)), zorder=2)
        ax.annotate("population benchmark", (NS[-1], popk["bench"]), xytext=(-2, 3), textcoords="offset points",
                    fontsize=6.5, color=P.CAT[6], ha="right", va="bottom", bbox=box, zorder=4)
    log_axis(ax)
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.28 * float(aggB.RV_hi.max()))
    ax.legend(loc="upper right", fontsize=6.3)
    P.savefig(fig, "exp2_text_rv_benchmark.pdf")

    # ---- power (n = 1000)
    dfC = pd.read_csv(RESULTS_DIR / "exp2_text_power_raw.csv")
    if "gamma" not in dfC:
        dfC = dfC.rename(columns={"beta": "gamma"})
    if "d0" in dfC.columns:
        dfC = add_hybrid_bench(dfC)
    if f"rho_{PRIMARY}" in dfC.columns:
        P1, S1 = PRIMARY, SECONDARY
        bound_sec = np.array([pop[f"lam{l}_{S1}"]["bound"] for l in dfC.lam])
        dfC["naive"] = naive_reject(dfC[f"rho_{P1}"], dfC[f"se_{P1}"])
        dfC["robust_bench"] = robust_reject(dfC[f"rho_{P1}"], dfC[f"se_{P1}"], dfC[f"bench_{P1}"].values)
        dfC["robust_bound"] = robust_reject(dfC[f"rho_{P1}"], dfC[f"se_{P1}"], dfC.bound_pop.values)
        dfC[f"naive_{S1}"] = naive_reject(dfC[f"rho_{S1}"], dfC[f"se_{S1}"])
        dfC[f"robust_bench_{S1}"] = robust_reject(dfC[f"rho_{S1}"], dfC[f"se_{S1}"], dfC[f"bench_{S1}"].values)
        dfC[f"robust_bound_{S1}"] = robust_reject(dfC[f"rho_{S1}"], dfC[f"se_{S1}"], bound_sec)
        dfC.to_csv(RESULTS_DIR / "exp2_text_power_decisions.csv", index=False)
        aggC = dfC.groupby(["lam", "gamma"]).agg(**{"naive": ("naive", "mean"), "robust_bench": ("robust_bench", "mean"), "robust_bound": ("robust_bound", "mean"),
                                                    f"naive_{S1}": (f"naive_{S1}", "mean"), f"robust_bench_{S1}": (f"robust_bench_{S1}", "mean"),
                                                    f"robust_bound_{S1}": (f"robust_bound_{S1}", "mean"),
                                                    "bench": (f"bench_{P1}", "mean"), "rho": (f"rho_{P1}", "mean")}).reset_index()
        aggC.to_csv(RESULTS_DIR / "exp2_text_power.csv", index=False)
        summary["power"] = aggC.to_dict(orient="records")
        summary["n_rep_per_setting"]["power"] = int(dfC.groupby(["lam", "gamma"]).size().min())
    else:   # the raw file holds the aggregated table (per-replication draws of the power phase were not kept)
        aggC = dfC.copy()
    fig, axes = plt.subplots(1, 2, figsize=P.DOUBLE, sharey=True)
    for ax, lam, title in zip(axes, (0.0, 1.0), (r"$\lambda=0$: insufficient $\psi$, no bias", r"$\lambda=1$: insufficient $\psi$, bias at the bound")):
        a = aggC[aggC.lam == lam]
        ax.plot(a.gamma, a.naive, marker="o", color=P.CAT[1], label="standard test")
        ax.plot(a.gamma, a.robust_bound, marker="o", color=P.CAT[6], label=r"robust, $B=$ oracle bound")
        ax.axhline(ALPHA, color=P.MUTED, linewidth=0.8, linestyle="--", zorder=1)
        ax.set_title(title)
        ax.set_xlabel(r"effect size $\gamma$ ($Y \leftarrow Y + \gamma X$)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("rejection rate")
    axes[0].legend(loc="center right")
    P.savefig(fig, "exp2_text_power.pdf")
    (RESULTS_DIR / "summary_exp2.json").write_text(json.dumps(summary, indent=1, default=float))
    print("plots written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_rep", nargs="?", type=int, default=100)
    ap.add_argument("--only", default="pop,typeI,bench,power,plots")
    args = ap.parse_args()
    phases = args.only.split(",")
    t0 = time.time()
    if "pop" in phases:
        print(json.dumps(get_pop(), indent=1), flush=True)
    if "typeI" in phases:
        phase_typeI(args.n_rep)
    if "bench" in phases:
        phase_bench(args.n_rep)
    if "power" in phases:
        phase_power(args.n_rep)
    if "plots" in phases:
        phase_plots()
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
