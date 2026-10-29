"""Experiment 4, ground-truth tolerance, phase 2: identify the sensitivity parameters of every view psi from the M regenerated
draws of exp4_regen.py and check the robust test with the true tolerance.

By the proposition "Identification with replicates", with X_{im}, m = 1..M, conditionally i.i.d. given the post Z_i,
  Var(E[X|Z])            = between-post component of the balanced one-way ANOVA, (M Var(Xbar_i) - MSW) / M,
  Cov(E[X|Z], E[Y|Z])    = the same for the cross products,
  Var(E[X|psi(Z)])       = Cov_n(mhat_X(psi(Z_i)), X_i) with the cross-fitted fit mhat_X: unbiased to first order, without the
                           d/n inflation of Var(mhat_X) and of the cross-fitted MSE (theorem "Total adjustment error"),
  Cov(E[X|psi], E[Y|psi]) = (Cov_n(mhat_X, Y) + Cov_n(X, mhat_Y)) / 2,
so that sigma^2_psi(X) = Var(X) - Var(E[X|psi]), E[Delta_X^2] = Var(E[X|Z]) - Var(E[X|psi]), R^2_psi(X) = E[Delta_X^2] / sigma^2_psi(X),
Cov(Delta_X, Delta_Y) = Cov(E[X|Z], E[Y|Z]) - Cov(E[X|psi], E[Y|psi]), lambda, B* = sqrt(R^2_psi(X) R^2_psi(Y)), and the predicted
rho_psi = lambda B*. Also reported: the realized total-error version B_f of the theorem (cross-fitted MSEs in place of the
population residual variances), the within-post covariance (estimates tau_id, zero under H0), bootstrap intervals over posts.

Datasets: every draw m is a dataset (rho_hat^(m), se^(m), the standard test, the robust test at B*, at B*/2 and at the true bias
|lambda| B*, the oracle boundary where the level must equal alpha); R resampled
datasets pick one comment draw and one summary draw per post, which are exact null draws given the posts, for the rejection
rates; the observed dataset is draw 0 (--observed 0) or the HF-generated data of exp4_local.py (--observed hf), and it defines
the cross-fitted sentiment direction as in exp4_local.py.
Outputs: results/exp4_truth.csv (one row per view), exp4_truth_datasets.csv, exp4_truth.json, a diagnostic figure in results/preview;
--write adds the \\resFourTruth* block to paper/results_macros.tex. The paper figures (fig:exp4, fig:exp4identity) are made by fig_exp4.py.
The paper uses --observed hf (the HF-generated data of exp4_local.py as the observed dataset, the vLLM draws as replicates).
usage: python experiments/exp4_truth.py ~/.cache/ci_text/llm/regen_qwen3-1.7b_n3000_M20.jsonl --observed hf --write
       [--dims 64,128,256,384] [--views E1,E2,mean_last,mean_mid,tok_last,tok_mid] [--resample 200] [--boot 200] [--out DIR] [--no-rule] [--regressor ols|mlp]
Regressor (7 Oct 2026): the paper uses least squares. --regressor mlp fits the small networks of citext/nn.py instead (the configuration
chosen per fold on the observed X and Y is reused for the draws and the null datasets) and writes exp4_truth_nn.csv, exp4_truth_nn_datasets.csv
and exp4_truth_nn.json; its --write adds a separate block of \\resFourTruthNn* macros (not used in the paper), with the least squares values of
exp4_truth.csv as ...Ls macros and the network residuals at the least squares tolerances as RejBls*/RejLamBls*. A network variant of the
paper was tried and dropped on 7 Oct 2026 (the identification formulas assume projection fits; see the Nn block). Slurm job: exp4_nn.sbatch.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("CI_TEXT_CODE_DIR", HERE.parent)); sys.path.insert(0, str(CODE))
from citext.estimators import fit_predict_crossfit, crossfit_folds, residual_correlation, critical_value, rv_alpha, config_feasible
from citext.nn import crossfit_nn
from citext.paths import RESULTS_DIR, CACHE_DIR, FIG_DIR, PAPER_DIR
from citext.text_semi import E1_NAME, E2_NAME, embed_texts
LLM = CACHE_DIR / "llm"; SOURCE = LLM / "gen_claude-haiku-4-5_n3000.jsonl"
ALPHA, SEED, K = 0.05, 0, 5
LABEL_SCORE = {"POSITIVE": 1.0, "NEUTRAL": 0.0, "NEGATIVE": -1.0}
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ estimators (pure numpy; checked by test_exp4_truth.py)
def cov(a, b): return float(np.mean((a - a.mean()) * (b - b.mean())))

def z_components(Xm, Ym):
    """Balanced one-way ANOVA over posts; Xm, Ym of shape (n, M)."""
    n, M = Xm.shape; xb, yb = Xm.mean(1), Ym.mean(1); dwx, dwy = Xm - xb[:, None], Ym - yb[:, None]
    msw_x, msw_y, msw_xy = np.sum(dwx**2) / (n * (M - 1)), np.sum(dwy**2) / (n * (M - 1)), np.sum(dwx * dwy) / (n * (M - 1))
    msb_x, msb_y, msb_xy = M * np.var(xb, ddof=1), M * np.var(yb, ddof=1), M * np.cov(xb, yb, ddof=1)[0, 1]
    within_xy = np.sum(dwx * dwy, 1) / (M - 1)   # per-post unbiased within-post covariance; its mean estimates tau_id
    return {"var_X": float(np.mean(np.var(Xm, axis=0, ddof=1))), "var_Y": float(np.mean(np.var(Ym, axis=0, ddof=1))),
            "var_EX_Z": float((msb_x - msw_x) / M), "var_EY_Z": float((msb_y - msw_y) / M), "cov_EXEY_Z": float((msb_xy - msw_xy) / M),
            "sigma2_id_X": float(msw_x), "sigma2_id_Y": float(msw_y), "tau_id": float(msw_xy), "tau_id_se": float(np.std(within_xy, ddof=1) / np.sqrt(n))}

def psi_components(Xm, Ym, mX, mY):
    """Population quantities of a view from the cross-fitted fits mX, mY of shape (n, M)."""
    M = Xm.shape[1]
    return {"var_EX_psi": float(np.mean([cov(mX[:, m], Xm[:, m]) for m in range(M)])), "var_EY_psi": float(np.mean([cov(mY[:, m], Ym[:, m]) for m in range(M)])),
            "cov_EXEY_psi": float(np.mean([(cov(mX[:, m], Ym[:, m]) + cov(Xm[:, m], mY[:, m])) / 2 for m in range(M)])),
            "mse_cf_X": float(np.mean((Xm - mX)**2)), "mse_cf_Y": float(np.mean((Ym - mY)**2))}

def sensitivity(zc, pc):
    """R^2_psi(X), R^2_psi(Y), lambda, B*, the predicted rho_psi, and the realized total-error version."""
    s2X, s2Y = zc["var_X"] - pc["var_EX_psi"], zc["var_Y"] - pc["var_EY_psi"]
    d2X, d2Y = max(zc["var_EX_Z"] - pc["var_EX_psi"], 0.0), max(zc["var_EY_Z"] - pc["var_EY_psi"], 0.0)
    r2X, r2Y = d2X / s2X, d2Y / s2Y; cd = zc["cov_EXEY_Z"] - pc["cov_EXEY_psi"]
    lam = cd / np.sqrt(d2X * d2Y) if d2X > 0 and d2Y > 0 else float("nan"); B = float(np.sqrt(r2X * r2Y))
    estX, estY = max(pc["mse_cf_X"] - s2X, 0.0), max(pc["mse_cf_Y"] - s2Y, 0.0)   # mean squared estimation error of the two regressions
    r2fX, r2fY = (d2X + estX) / pc["mse_cf_X"], (d2Y + estY) / pc["mse_cf_Y"]
    return {"sigma2_psi_X": s2X, "sigma2_psi_Y": s2Y, "delta2_X": d2X, "delta2_Y": d2Y, "R2_psi_X": r2X, "R2_psi_Y": r2Y, "cov_delta": cd,
            "lambda": float(lam), "B_star": B, "pred_rho": float(lam * B) if np.isfinite(lam) else float("nan"),
            "mse_est_X": estX, "mse_est_Y": estY, "R2_f_X": r2fX, "R2_f_Y": r2fY, "B_total": float(np.sqrt(r2fX * r2fY)),
            "R2_view_X_pop": pc["var_EX_psi"] / zc["var_X"], "R2_view_Y_pop": pc["var_EY_psi"] / zc["var_Y"],
            "R2_view_X_cf": 1 - pc["mse_cf_X"] / zc["var_X"], "R2_view_Y_cf": 1 - pc["mse_cf_Y"] / zc["var_Y"],
            "share_X_Z": zc["var_EX_Z"] / zc["var_X"], "share_Y_Z": zc["var_EY_Z"] / zc["var_Y"]}

def bootstrap(Xm, Ym, mX, mY, n_boot, rng):
    """Percentile intervals over posts for B*, lambda and the partial R^2 values, with the fits held fixed."""
    n = Xm.shape[0]; out = []
    for _ in range(n_boot):
        ii = rng.integers(n, size=n); s = sensitivity(z_components(Xm[ii], Ym[ii]), psi_components(Xm[ii], Ym[ii], mX[ii], mY[ii]))
        out.append((s["B_star"], s["lambda"], s["R2_psi_X"], s["R2_psi_Y"]))
    lo, hi = np.nanpercentile(np.array(out, float), [2.5, 97.5], axis=0)
    return {"B_star_lo": lo[0], "B_star_hi": hi[0], "lambda_lo": lo[1], "lambda_hi": hi[1], "R2_psi_X_lo": lo[2], "R2_psi_X_hi": hi[2], "R2_psi_Y_lo": lo[3], "R2_psi_Y_hi": hi[3]}

def robust_p(rho, se, B): return float(norm.sf((abs(rho) - B) / se) + norm.sf((abs(rho) + B) / se))

def dataset_stats(rX, rY, B, B_true=None):
    """Standard test and robust tests at the tolerances B* (= B), B*/2, and the true bias |lambda| B* (= B_true, the oracle boundary)."""
    rho, se = residual_correlation(rX, rY); Bt = B if B_true is None or not np.isfinite(B_true) else B_true
    return {"rho": rho, "se": se, "p": float(2 * norm.sf(abs(rho) / se)), "RV_alpha": float(rv_alpha(rho, se, ALPHA)),
            "reject_std": bool(abs(rho) > critical_value(0.0, se, ALPHA)), "reject_B": bool(abs(rho) > B + critical_value(B, se, ALPHA)),
            "reject_Bhalf": bool(abs(rho) > B / 2 + critical_value(B / 2, se, ALPHA)), "p_B": robust_p(rho, se, B),
            "reject_lamB": bool(abs(rho) > Bt + critical_value(Bt, se, ALPHA)), "p_lamB": robust_p(rho, se, Bt)}

# ------------------------------------------------------------------ data
def load_draws(path, M_use=None):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]; by = {}
    for r in rows: by.setdefault(r["idx"], {})[r["m"]] = r
    M = (max(r["m"] for r in rows) + 1) if M_use is None else M_use
    # every draw needs both texts; a label is needed only for draw 0, which can serve as the observed dataset (the direction fit)
    keep = sorted(i for i, d in by.items() if all(m in d and d[m]["X"] and d[m]["Y0"] for m in range(M)) and d[0]["X_label"])
    return keep, M, by

def load_observed(kind, keep, by, n_src):
    if kind == "0": return {i: by[i][0] for i in keep}
    rows = [json.loads(l) for l in (LLM / f"gen_qwen3-1.7b_n{n_src}.jsonl").read_text().splitlines() if l.strip()]
    d = {r["idx"]: r for r in rows if r["X_label"] and r["X"] and r["Y0"]}; return {i: d[i] for i in keep if i in d}

def fold_directions(E_obs, labels, folds):
    """Cross-fitted ridge direction from E1(comment) to the self-reported label, as in exp4_local.sentiment_axis."""
    from sklearn.linear_model import RidgeCV
    y = np.array([LABEL_SCORE[l] for l in labels]); return [RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(E_obs[tr], y[tr]) for tr, te in folds]

def apply_directions(models, folds, E):
    """E of shape (n, d) or (n, M, d): fold k's direction scores the rows of fold k, for every draw."""
    out = np.empty(E.shape[:-1])
    for r, (tr, te) in zip(models, folds): out[te] = r.predict(E[te].reshape(-1, E.shape[-1])).reshape(E[te].shape[:-1])
    return out

def build_views(keep, E1Z, E2Z, hidden, names, dims, rule):
    views = {}
    if "E1" in names: views["E1"] = E1Z
    if "E2" in names: views["E2"] = E2Z
    hn = [v for v in names if v not in ("E1", "E2")]
    if hn and hidden is not None and hidden.exists():
        from sklearn.decomposition import PCA
        H = np.load(hidden); pos = {int(i): j for j, i in enumerate(H["idx"])}; sel = np.array([pos[i] for i in keep])
        for name in hn:
            Hv = H[name][sel]; dmax = min(max(dims), Hv.shape[0] - 1, Hv.shape[1]); T = PCA(dmax, random_state=SEED).fit_transform(Hv)
            for d in dims:
                if d <= dmax: views[f"{name}:{d}"] = T[:, :d]
    elif hn: log(f"hidden states not found at {hidden}: only E1/E2")
    n = len(keep); out = {}
    for k, v in views.items():
        if rule and not config_feasible(n, v.shape[1], K): log(f"  {k} (d={v.shape[1]}): not reported under the rule"); continue
        out[k] = v
    return out

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("regen"); ap.add_argument("--observed", choices=("0", "hf"), default="0"); ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--views", default="E1,E2,mean_last,mean_mid,tok_last,tok_mid"); ap.add_argument("--dims", default="64,128,256,384")
    ap.add_argument("--resample", type=int, default=200); ap.add_argument("--boot", type=int, default=200); ap.add_argument("--no-rule", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR)); ap.add_argument("--figdir", default=str(RESULTS_DIR / "preview"), help="diagnostic two-panel figure; the paper figures come from fig_exp4.py"); ap.add_argument("--write", action="store_true"); ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--regressor", choices=("ols", "mlp"), default="ols", help="least squares (result files without suffix; the comparison) or the small networks of citext.nn (files exp4_truth_nn*; the paper's Experiment 3)")
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); figdir = Path(a.figdir).expanduser(); figdir.mkdir(parents=True, exist_ok=True)
    sfx = "" if a.regressor == "ols" else "_nn"; chosen_all = {}
    regen = Path(a.regen).expanduser(); stem = regen.stem; n_src = int(re.search(r"_n(\d+)_", stem).group(1)); dims = [int(d) for d in a.dims.split(",")]
    keep, M, by = load_draws(regen, a.M); obs = load_observed(a.observed, keep, by, n_src); keep = [i for i in keep if i in obs]; n = len(keep)
    log(f"{n} posts with all {M} draws labelled; observed dataset: {'draw 0' if a.observed == '0' else 'HF-generated data'}")
    src = {r["idx"]: r["Z"] for r in (json.loads(l) for l in SOURCE.read_text().splitlines() if l.strip())}; Z = [src[i] for i in keep]
    labels = [obs[i]["X_label"] for i in keep]; ck = f"{stem}_{n}"
    E1Z, E2Z = embed_texts(Z, E1_NAME, f"{ck}_Z"), embed_texts(Z, E2_NAME, f"{ck}_Z")
    E1Xo = embed_texts([obs[i]["X"] for i in keep], E1_NAME, f"{ck}_Xobs{a.observed}"); E1Yo = embed_texts([obs[i]["Y0"] for i in keep], E1_NAME, f"{ck}_Yobs{a.observed}")
    E1Xd = embed_texts([by[i][m]["X"] for i in keep for m in range(M)], E1_NAME, f"{ck}_Xdraws").reshape(n, M, -1)
    E1Yd = embed_texts([by[i][m]["Y0"] for i in keep for m in range(M)], E1_NAME, f"{ck}_Ydraws").reshape(n, M, -1)
    folds = crossfit_folds(n, K, SEED); dirs = fold_directions(E1Xo, labels, folds)
    Xo, Yo = apply_directions(dirs, folds, E1Xo), apply_directions(dirs, folds, E1Yo); Xm, Ym = apply_directions(dirs, folds, E1Xd), apply_directions(dirs, folds, E1Yd)
    zc = z_components(Xm, Ym)
    log(f"post shares: Var(E[X|Z])/Var X = {zc['var_EX_Z']/zc['var_X']:.3f}, Var(E[Y|Z])/Var Y = {zc['var_EY_Z']/zc['var_Y']:.3f}, "
        f"corr(E[X|Z],E[Y|Z]) = {zc['cov_EXEY_Z']/np.sqrt(zc['var_EX_Z']*zc['var_EY_Z']):.3f}; tau_id = {zc['tau_id']:.4f} (se {zc['tau_id_se']:.4f}); "
        f"unadjusted corr: observed {np.corrcoef(Xo, Yo)[0,1]:.3f}, draws {np.mean([np.corrcoef(Xm[:,m], Ym[:,m])[0,1] for m in range(M)]):.3f}")
    rng = np.random.default_rng(SEED); R = a.resample; ar = np.arange(n); mx_, my_ = rng.integers(M, size=(R, n)), rng.integers(M, size=(R, n))
    Xres, Yres = np.stack([Xm[ar, mx_[r]] for r in range(R)], 1), np.stack([Ym[ar, my_[r]] for r in range(R)], 1)
    views = build_views(keep, E1Z, E2Z, LLM / f"hidden_qwen3-1.7b_n{n_src}.npz", a.views.split(","), dims, not a.no_rule)
    rows_v, rows_d = [], []
    for name, Psi in views.items():
        T = np.hstack([Xm, Ym, Xo[:, None], Yo[:, None], Xres, Yres])
        if a.regressor == "ols": That = fit_predict_crossfit(Psi, T, K, SEED, "ols", folds); fit_info = {}
        else:   # networks: the configuration is chosen per fold on the observed X and Y (columns 2M and 2M + 1) and reused for the draws and the null datasets
            That, nfo = crossfit_nn(Psi, T, folds, select=[2 * M] * M + [2 * M + 1] * M + [None, None] + [2 * M] * R + [2 * M + 1] * R)
            chosen_all[name] = {"X": nfo["chosen"][2 * M], "Y": nfo["chosen"][2 * M + 1]}; names = lambda j: "|".join(sorted({c for nm in nfo["chosen"][j] for c in nm.split("+")}))
            fit_info = {"configs_X": names(2 * M), "configs_Y": names(2 * M + 1), "val_r2": nfo["val_r2"], "iters": nfo["iters"]}
        mX, mY, mXo, mYo = That[:, :M], That[:, M:2 * M], That[:, 2 * M], That[:, 2 * M + 1]; mXr, mYr = That[:, 2 * M + 2:2 * M + 2 + R], That[:, 2 * M + 2 + R:]
        pc = psi_components(Xm, Ym, mX, mY); s = sensitivity(zc, pc); B = s["B_star"]; bo = bootstrap(Xm, Ym, mX, mY, a.boot, np.random.default_rng(SEED + 1))
        Bt = abs(s["pred_rho"]); ob = dataset_stats(Xo - mXo, Yo - mYo, B, Bt); dr = [dataset_stats(Xm[:, m] - mX[:, m], Ym[:, m] - mY[:, m], B, Bt) for m in range(M)]
        rs = [dataset_stats(Xres[:, r] - mXr[:, r], Yres[:, r] - mYr[:, r], B, Bt) for r in range(R)]
        rows_d += [{"psi": name, "kind": "draw", "k": m, **d} for m, d in enumerate(dr)] + [{"psi": name, "kind": "resampled", "k": r, **d} for r, d in enumerate(rs)]
        rate = lambda L, key: float(np.mean([d[key] for d in L])); rhos_d, rhos_r = [d["rho"] for d in dr], [d["rho"] for d in rs]
        row = {"psi": name, "d": Psi.shape[1], "n": n, "M": M, "R": R, "regressor": a.regressor, **{f"obs_{k}": v for k, v in ob.items()},
               "obs_R2_X": float(1 - np.mean((Xo - mXo) ** 2) / np.var(Xo)), "obs_R2_Y": float(1 - np.mean((Yo - mYo) ** 2) / np.var(Yo)), **fit_info, **pc, **s, **bo,
               "rho_draws_mean": float(np.mean(rhos_d)), "rho_draws_sd": float(np.std(rhos_d, ddof=1)) if M > 1 else float("nan"), "se_draws_mean": float(np.mean([d["se"] for d in dr])),
               "rho_res_mean": float(np.mean(rhos_r)), "rho_res_sd": float(np.std(rhos_r, ddof=1)), "se_res_mean": float(np.mean([d["se"] for d in rs])),
               "rej_std_draws": rate(dr, "reject_std"), "rej_B_draws": rate(dr, "reject_B"), "rej_Bhalf_draws": rate(dr, "reject_Bhalf"),
               "rej_std_res": rate(rs, "reject_std"), "rej_B_res": rate(rs, "reject_B"), "rej_Bhalf_res": rate(rs, "reject_Bhalf"), "p_B_res_mean": rate(rs, "p_B"),
               "rej_lamB_draws": rate(dr, "reject_lamB"), "rej_lamB_res": rate(rs, "reject_lamB"), "p_lamB_res_mean": rate(rs, "p_lamB")}
        rows_v.append(row)
        log(f"{name} (d={Psi.shape[1]}): rho obs {ob['rho']:.3f} (RV_a {ob['RV_alpha']:.3f}, R2 obs {row['obs_R2_X']:.3f}/{row['obs_R2_Y']:.3f}), draws {row['rho_draws_mean']:.3f}, resampled {row['rho_res_mean']:.3f}+-{row['rho_res_sd']:.3f} (se {row['se_res_mean']:.3f}) | "
            f"R2_psi X {s['R2_psi_X']:.3f} Y {s['R2_psi_Y']:.3f} lambda {s['lambda']:.3f} B* {B:.3f} [{bo['B_star_lo']:.3f},{bo['B_star_hi']:.3f}] B_f {s['B_total']:.3f} pred {s['pred_rho']:.3f} | "
            f"reject std {row['rej_std_res']:.2f} B* {row['rej_B_res']:.2f} B*/2 {row['rej_Bhalf_res']:.2f} lamB* {row['rej_lamB_res']:.2f} (draws {row['rej_lamB_draws']:.2f}) | view R2 pop {s['R2_view_X_pop']:.3f}/{s['R2_view_Y_pop']:.3f} cf {s['R2_view_X_cf']:.3f}/{s['R2_view_Y_cf']:.3f}")
    V = pd.DataFrame(rows_v); V.to_csv(out / f"exp4_truth{sfx}.csv", index=False); pd.DataFrame(rows_d).to_csv(out / f"exp4_truth{sfx}_datasets.csv", index=False)
    summ = {"regen": str(regen), "observed": a.observed, "regressor": a.regressor, "chosen": chosen_all, "n": n, "M": M, "R": R, "views": list(views), **zc, "unadjusted_obs": float(np.corrcoef(Xo, Yo)[0, 1]),
            "unadjusted_draws": float(np.mean([np.corrcoef(Xm[:, m], Ym[:, m])[0, 1] for m in range(M)])), "labels_obs": pd.Series(labels).value_counts().to_dict()}
    json.dump(summ, open(out / f"exp4_truth{sfx}.json", "w"), indent=1); log(f"written to {out} (files exp4_truth{sfx}*)")
    if not a.no_plot: plot(V, summ, figdir, sfx)
    macros(V, summ, write=a.write, regressor=a.regressor, out=out)

# ------------------------------------------------------------------ figure
def plot(V, summ, figdir, sfx=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from citext import plotting as P
    P.setup(); fig, (ax, bx) = plt.subplots(1, 2, figsize=P.DOUBLE, gridspec_kw={"wspace": 0.35})
    fam = {"mean_last": ("mean, last layer", P.CAT[0], "-"), "mean_mid": ("mean, middle layer", P.CAT[0], "--"), "tok_last": ("last token, last layer", P.CAT[6], "-"), "tok_mid": ("last token, middle layer", P.CAT[6], "--"), "E1": ("$E_1$", P.CAT[1], "-"), "E2": ("$E_2$", P.CAT[2], "-")}
    V = V.copy(); V["fam"] = V.psi.str.split(":").str[0]
    # left: observed rho_psi against the identified bound B* (hollow) and the prediction lambda B* (filled), one marker per view
    lim = float(max(V.B_star.max(), V.rho_res_mean.abs().max(), 0.05)) * 1.1
    ax.plot([0, lim], [0, lim], color=P.AXIS, lw=0.8)
    for f, (lab, col, ls) in fam.items():
        d = V[V.fam == f]
        if not len(d): continue
        y = d.rho_res_mean.abs(); yerr = 1.96 * d.rho_res_sd
        ax.errorbar(d.B_star, y, yerr=yerr, fmt="o", mfc="white", color=col, ms=4, capsize=1.5, lw=0.8, label=lab if f in ("E1", "E2") else None)
        ax.errorbar(d.pred_rho.abs(), y, yerr=yerr, fmt="o", color=col, ms=4, capsize=1.5, lw=0.8)
        for _, r in d.iterrows():
            if ":" in r.psi: ax.annotate(r.psi.split(":")[1], (r.B_star, abs(r.rho_res_mean)), xytext=(3, -2), textcoords="offset points", fontsize=5, color=col)
    ax.errorbar([], [], fmt="o", mfc="white", color=P.INK2, ms=4, label="$\\sqrt{R^2_\\psi(X) R^2_\\psi(Y)}$"); ax.errorbar([], [], fmt="o", color=P.INK2, ms=4, label="$\\lambda\\sqrt{R^2_\\psi(X) R^2_\\psi(Y)}$")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("identified bound and prediction"); ax.set_ylabel("$\\hat\\rho_\\psi$ under $H_0$"); ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    # right: rejection rates against the dimension, standard test and robust test at B* and B*/2
    for f in ("mean_last", "tok_last"):
        d = V[V.fam == f].copy()
        if not len(d): continue
        d["dim"] = d.psi.str.split(":").str[1].astype(int); d = d.sort_values("dim"); lab, col, _ = fam[f]
        bx.plot(d.dim, d.rej_std_res, color=col, ls="-", marker="o", ms=3, lw=1.1, label=f"{lab}, standard"); bx.plot(d.dim, d.rej_Bhalf_res, color=col, ls=":", marker="o", ms=3, lw=1.1, label=f"{lab}, tolerance $B^*/2$")
        bx.plot(d.dim, d.rej_lamB_res, color=col, ls="-.", marker="o", ms=3, lw=1.1, label=f"{lab}, tolerance $\\lambda B^*$")
        bx.plot(d.dim, d.rej_B_res, color=col, ls="--", marker="o", ms=3, lw=1.1, label=f"{lab}, tolerance $B^*$")
    for f, mk in (("E1", "s"), ("E2", "D")):
        d = V[V.fam == f]
        if not len(d): continue
        r = d.iloc[0]; lab, col, _ = fam[f]
        bx.plot([r.d], [r.rej_std_res], mk, color=col, ms=4.5, label=f"$\\psi = {lab[1:-1]}$"); bx.plot([r.d], [r.rej_B_res], mk, mfc="white", color=col, ms=4.5); bx.plot([r.d], [r.rej_Bhalf_res], mk, mfc="white", color=col, ms=3)
    bx.axhline(ALPHA, color=P.MUTED, ls=":", lw=0.9); bx.set_xscale("log", base=2); dims = sorted(set(int(p.split(":")[1]) for p in V.psi if ":" in p) | set(V[V.fam.isin(["E1", "E2"])].d))
    bx.set_xticks(dims); bx.set_xticklabels([str(x) for x in dims]); bx.minorticks_off(); bx.set_ylim(-0.02, 1.05); bx.set_xlabel("dimension of $\\psi(Z)$"); bx.set_ylabel("rejection rate under $H_0$")
    bx.legend(fontsize=5.2, frameon=False, loc="center right", handlelength=2.2)
    fig.savefig(figdir / f"exp4_truth{sfx}.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(figdir / f"exp4_truth{sfx}.png", dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig); log(f"figure: {figdir / f'exp4_truth{sfx}.pdf'}")

# ------------------------------------------------------------------ macros
def macros(V, summ, write, regressor="ols", out=None):
    """The \\resFourTruth* block. With regressor 'mlp' (the paper's) the least squares values of exp4_truth.csv, on the same folds and data,
    are added with the suffix Ls, and the network residuals tested at the least squares tolerances as RejBls (B*) and RejLamBls (|lambda| B*)."""
    f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"; pct = lambda v: f"{100*v:.0f}\\%"
    W = {64: "SixtyFour", 128: "OneTwentyEight", 256: "TwoFiftySix", 384: "ThreeEightyFour", 768: "SevenSixtyEight"}
    keys = {"MeanLast": "mean_last", "MeanMid": "mean_mid", "TokLast": "tok_last", "TokMid": "tok_mid"}
    M = {"resFourTruthM": str(summ["M"]), "resFourTruthN": f"{summ['n']:,}".replace(",", "{,}"), "resFourTruthR": str(summ["R"]), "resFourTruthShareXZ": pct(summ["var_EX_Z"] / summ["var_X"]), "resFourTruthShareYZ": pct(summ["var_EY_Z"] / summ["var_Y"]),
         "resFourTruthTauId": f"{summ['tau_id']:.4f}", "resFourTruthTauIdSe": f"{summ['tau_id_se']:.4f}", "resFourTruthMargNull": f3(summ["unadjusted_draws"]), "resFourTruthMargObs": f3(summ["unadjusted_obs"]),
         "resFourTruthRegressor": {"ols": "least squares", "mlp": "small neural networks"}[regressor]}
    def block(key, r, sfx=""):
        M[f"resFourTruthBstar{key}{sfx}"] = f3(r.B_star); M[f"resFourTruthBstarLo{key}{sfx}"] = f3(r.B_star_lo); M[f"resFourTruthBstarHi{key}{sfx}"] = f3(r.B_star_hi); M[f"resFourTruthBtotal{key}{sfx}"] = f3(r.B_total)
        M[f"resFourTruthLambda{key}{sfx}"] = f2(r["lambda"]); M[f"resFourTruthRtwoX{key}{sfx}"] = pct(r.R2_psi_X); M[f"resFourTruthRtwoY{key}{sfx}"] = pct(r.R2_psi_Y)
        M[f"resFourTruthPredRho{key}{sfx}"] = f3(abs(r.pred_rho)); M[f"resFourTruthRhoObs{key}{sfx}"] = f3(abs(r.obs_rho)); M[f"resFourTruthRhoMean{key}{sfx}"] = f3(abs(r.rho_res_mean)); M[f"resFourTruthRValphaObs{key}{sfx}"] = f3(r.obs_RV_alpha)
        M[f"resFourTruthRejStd{key}{sfx}"] = pct(r.rej_std_res); M[f"resFourTruthRejB{key}{sfx}"] = pct(r.rej_B_res); M[f"resFourTruthRejBhalf{key}{sfx}"] = pct(r.rej_Bhalf_res); M[f"resFourTruthRejLamB{key}{sfx}"] = pct(r.rej_lamB_res)
        M[f"resFourTruthViewRtwoXPop{key}{sfx}"] = pct(max(r.R2_view_X_pop, 0)); M[f"resFourTruthViewRtwoYPop{key}{sfx}"] = pct(max(r.R2_view_Y_pop, 0))
        if "obs_R2_X" in r.index: M[f"resFourTruthRtwoXObs{key}{sfx}"] = pct(max(r.obs_R2_X, 0)); M[f"resFourTruthRtwoYObs{key}{sfx}"] = pct(max(r.obs_R2_Y, 0))
    Vi = V.set_index("psi"); named = [("Eone", "E1"), ("Etwo", "E2")] + [(f"{key}{w}", f"{name}:{d}") for key, name in keys.items() for d, w in W.items()]
    for key, psi in named:
        if psi in Vi.index: block(key, Vi.loc[psi])
    hv = V[V.psi.str.contains(":")]
    if len(hv): M["resFourTruthBstarMinStates"] = f3(hv.B_star.min()); M["resFourTruthBstarMaxStates"] = f3(hv.B_star.max()); M["resFourTruthLambdaMinStates"] = f2(hv["lambda"].min()); M["resFourTruthLambdaMaxStates"] = f2(hv["lambda"].max())
    M["resFourTruthRejBMax"] = pct(V.rej_B_res.max()); M["resFourTruthRejBhalfMin"] = pct(V.rej_Bhalf_res.min()); M["resFourTruthRejBhalfMax"] = pct(V.rej_Bhalf_res.max()); M["resFourTruthRejLamBMin"] = pct(V.rej_lamB_res.min()); M["resFourTruthRejLamBMax"] = pct(V.rej_lamB_res.max()); M["resFourTruthRejLamBMean"] = pct(V.rej_lamB_res.mean()); M["resFourTruthRejLamBDrawsMean"] = pct(V.rej_lamB_draws.mean()); M["resFourTruthPredDiffMax"] = f3((V.pred_rho.abs() - V.rho_res_mean.abs()).abs().max())
    M["resFourTruthPredOvershootMin"] = f3((V.pred_rho.abs() - V.rho_res_mean.abs()).min()); M["resFourTruthPredOvershootMax"] = f3((V.pred_rho.abs() - V.rho_res_mean.abs()).max())
    M["resFourTruthBoundViolations"] = str(int((V.rho_res_mean.abs() > V.B_star).sum())); M["resFourTruthRhoObsMinAll"] = f3(V.obs_rho.abs().min()); M["resFourTruthRhoObsMaxAll"] = f3(V.obs_rho.abs().max())
    if len(hv): M["resFourTruthRhoObsMinStates"] = f3(hv.obs_rho.abs().min()); M["resFourTruthRhoObsMaxStates"] = f3(hv.obs_rho.abs().max())
    if regressor == "mlp" and out is not None and (out / "exp4_truth.csv").exists():   # least squares on the same folds and data (the comparison, suffix Ls), and the network residuals at its tolerances
        L = pd.read_csv(out / "exp4_truth.csv").set_index("psi"); common = [p for p in Vi.index if p in L.index]
        dp = out / "exp4_truth_nn_datasets.csv"; D = pd.read_csv(dp) if dp.exists() else pd.DataFrame(columns=["psi", "kind", "rho", "se"]); D = D[D.kind == "resampled"]
        for key, psi in named:
            if psi not in common: continue
            r = L.loc[psi]; block(key, r, "Ls"); d = D[D.psi == psi]
            if len(d):
                Bl, Btl = float(r.B_star), abs(float(r.pred_rho))
                M[f"resFourTruthRejBls{key}"] = pct(np.mean([abs(rho) > Bl + critical_value(Bl, se, ALPHA) for rho, se in zip(d.rho, d.se)]))
                M[f"resFourTruthRejLamBls{key}"] = pct(np.mean([abs(rho) > Btl + critical_value(Btl, se, ALPHA) for rho, se in zip(d.rho, d.se)]))
        if common:
            Lc, Vc = L.loc[common], Vi.loc[common]; diff = Vc.obs_rho.abs() - Lc.obs_rho.abs(); M["resFourTruthRhoObsDiffMinLs"] = f3(diff.min()); M["resFourTruthRhoObsDiffMaxLs"] = f3(diff.max())
            dres = Vc.rho_res_mean.abs() - Lc.rho_res_mean.abs(); M["resFourTruthRhoMeanDiffMinLs"] = f3(dres.min()); M["resFourTruthRhoMeanDiffMaxLs"] = f3(dres.max())
            M["resFourTruthBstarMinLs"] = f3(Lc.B_star.min()); M["resFourTruthBstarMaxLs"] = f3(Lc.B_star.max()); M["resFourTruthLambdaMinLs"] = f2(Lc["lambda"].min()); M["resFourTruthLambdaMaxLs"] = f2(Lc["lambda"].max())
            M["resFourTruthRejLamBMeanLs"] = pct(Lc.rej_lamB_res.mean()); M["resFourTruthRejLamBMinLs"] = pct(Lc.rej_lamB_res.min()); M["resFourTruthRejLamBMaxLs"] = pct(Lc.rej_lamB_res.max()); M["resFourTruthRejBhalfMinLs"] = pct(Lc.rej_Bhalf_res.min())
            M["resFourTruthPredDiffMaxLs"] = f3((Lc.pred_rho.abs() - Lc.rho_res_mean.abs()).abs().max()); M["resFourTruthBstarMinCommon"] = f3(Vc.B_star.min()); M["resFourTruthBstarMaxCommon"] = f3(Vc.B_star.max())
            M["resFourTruthCommonViews"] = str(len(common))
    lines = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()); print(lines)
    if write:
        MAC = PAPER_DIR / "results_macros.tex"; s = MAC.read_text(); base = "% ---------- Experiment 4, ground-truth tolerance"
        if regressor == "mlp":   # the network comparison is not in the paper: its block carries the infix Nn
            lines = "\n".join(f"\\newcommand{{\\{k.replace('resFourTruth', 'resFourTruthNn', 1)}}}{{{v}}}" for k, v in M.items()); head = base + ", NETWORK COMPARISON (exp4_truth.py --regressor mlp; not used in the paper) ----------"
            drop = lambda l: re.match(r"\\newcommand\{\\resFourTruthNn", l) is not None or l.startswith(head)
        else:
            head = base + " ----------"; drop = lambda l: re.match(r"\\newcommand\{\\resFourTruth(?!Nn)", l) is not None or (l.startswith(base) and "NETWORK" not in l)
        keep = [l for l in s.splitlines() if not drop(l)]
        MAC.write_text("\n".join(keep).rstrip("\n") + "\n" + head + "\n" + lines + "\n"); print("written to", MAC)

if __name__ == "__main__":
    main()
