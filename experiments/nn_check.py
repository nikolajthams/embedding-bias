"""Small neural networks in place of the linear regressions (author's request, 6 Oct 2026).

SUPERSEDED on 7 Oct 2026: the recipe moved to citext/nn.py, and the pipeline scripts exp4_truth.py, exp4_power.py and app_cmv_deep.py
run it with `--regressor mlp` (result files *_nn.*), which is what the paper reports for Experiment 3 and the application. This script
is kept for the record (its exp3 phase belongs to the removed Claude experiment); its \\resNN* block stays in results_macros.tex only
because struck text of the appendix still refers to it.

Experiments 3 and 4 and the application are not linear by construction, so this script re-runs their residual regressions
with small multilayer perceptrons and reports rho_hat, the cross-fitted R^2, the benchmark and, in Experiment 4, the sensitivity
parameters identified from the replicates, next to the linear values on the same folds.

Regressor: sklearn MLPRegressor (ReLU, Adam, learning rate 1e-3, batches of 200, standardized inputs, early stopping on 15% of the
training fold with a patience of 20 epochs, at most 500 epochs). Within each training fold the width, depth and weight decay are
chosen among CONFIGS by the validation score of the early-stopping split, so the test fold enters no choice; the prediction is the
mean over SEEDS networks (different initializations and validation splits, each with its own choice). sklearn divides the L2 term by
the number of rows, so the penalties 10 and 100 are moderate: tuned on the Experiment 3 data (scratchpad nn/tune.py, 6 Oct 2026),
where the default penalty 1e-4 overfits to a negative out-of-sample R^2 and these settings match or slightly beat ridge regression.
One network per target and fold (no multi-task fitting), with seeds that differ between the X and the Y regression, so that the two
fits share no initialization; a shared seed correlates the two prediction errors and inflates rho_hat. The identification of Experiment 4 uses the same estimators as exp4_truth.py with the network fits in place of
least squares, plus the Cauchy-Schwarz version Cov(mhat, X)^2 / Var(mhat), a lower bound on Var(E[X | psi(Z)]) for any fit.

Phases: exp3 -> results/nn_check_exp3.csv, nn_check_exp3_curve.csv   (full sample H0/H1; benchmark E1 -> E2; E1 learning curve under H0)
        exp4 -> results/nn_check_exp4.csv, nn_check_exp4_datasets.csv (observed data; E1, E2, principal components and the full
                hidden states of the generator; identification from the replicates; level on resampled null datasets; the
                configuration chosen per fold on the observed X and Y is reused for the replicate draws and the null datasets)
        app  -> results/nn_check_app.csv, nn_check_app_bench.csv      (six main pairs; all embeddings for the two score pairs, E1 for the rest)
        macros -> \\resNN* block of paper/results_macros.tex (--write)
usage: python experiments/nn_check.py [--only exp3,exp4,app,macros] [--configs full|small] [--n-sub 20] [--resample 50] [--smoke] [--write]
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("CI_TEXT_CODE_DIR", HERE.parent)); sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "experiments"))
from citext.estimators import fit_predict_crossfit, crossfit_folds, residual_correlation, rv_alpha, critical_value, config_feasible  # noqa: E402
from citext.paths import RESULTS_DIR, CACHE_DIR, PAPER_DIR  # noqa: E402
from citext.text_semi import E1_NAME, E2_NAME, embed_texts  # noqa: E402
OUT = Path(os.environ.get("NN_OUT", RESULTS_DIR)); OUT.mkdir(parents=True, exist_ok=True)   # result files (default: code/results)
ALPHA, SEED, K = 0.05, 0, 5
from citext.nn import CONFIGS, SEEDS, mlp, crossfit_nn, cfg_summary  # noqa: E402   (the recipe, since 7 Oct 2026 in citext/nn.py)
REGEN = CACHE_DIR / "llm" / "regen_qwen3-1.7b_n3000_M20.jsonl"; GEN3 = CACHE_DIR / "llm" / "gen_claude-haiku-4-5_n3000.jsonl"
EXP4_VIEWS = ["E1", "E2", "mean_last:64", "mean_last:256", "mean_last:384", "mean_last:full", "tok_last:384", "tok_last:full"]
EXP4_RES_VIEWS = ["E1", "mean_last:384", "mean_last:full"]            # views with the level check on resampled null datasets
APP_VIEWS = ["E1", "E2", "[E1,E2]", "M", "[E1,M]", "L300"]; APP_PAIRS_ALL_VIEWS = [("politeness", "score"), ("log_length", "score")]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ helpers
def r2(y, m): return float(1 - np.mean((y - m) ** 2) / np.var(y))
def test(rA, rB):
    rho, se = residual_correlation(rA, rB)
    return {"rho": rho, "se": se, "RV_alpha": float(rv_alpha(rho, se, ALPHA)), "reject": bool(abs(rho) > critical_value(0.0, se, ALPHA))}
def tagged(d, tag): return {f"{k}_{tag}": v for k, v in d.items()}

# ------------------------------------------------------------------ Experiment 3
def phase_exp3(configs, n_sub, smoke):
    import exp3_llm as X3
    rows = X3.load_rows(GEN3); emb = X3.embeddings(rows, GEN3.stem); s = np.array([X3.LABEL_SCORE.get(r["X_label"], 0.0) for r in rows])
    A, B0, B1 = X3.sentiment_axis(emb[("E1", "X")], s, emb[("E1", "Y0")], emb[("E1", "Y1")]); n = len(A)
    psis = {"E1": emb[("E1", "Z")], "E2": emb[("E2", "Z")], "E1+E2": np.hstack([emb[("E1", "Z")], emb[("E2", "Z")]])}
    if smoke: psis = {"E1": psis["E1"], "E2": psis["E2"]}
    folds = crossfit_folds(n, K, SEED); out = []
    for hyp, B in (("H0", B0), ("H1", B1)):
        for pn, Psi in psis.items():
            lA, lB = fit_predict_crossfit(Psi, A, K, SEED, "ridge", folds), fit_predict_crossfit(Psi, B, K, SEED, "ridge", folds)
            mA, iA = crossfit_nn(Psi, A, folds, configs); mB, iB = crossfit_nn(Psi, B, folds, configs); rA, rB = A - mA, B - mB
            rec = {"hyp": hyp, "psi": pn, "d": Psi.shape[1], "n": n, **tagged(test(rA, rB), "nn"), **tagged(test(A - lA, B - lB), "lin"),
                   "R2_X_nn": r2(A, mA), "R2_Y_nn": r2(B, mB), "R2_X_lin": r2(A, lA), "R2_Y_lin": r2(B, lB),
                   "val_r2_X": iA["val_r2"], "val_r2_Y": iB["val_r2"], "iters": (iA["iters"] + iB["iters"]) / 2, "configs_X": cfg_summary(iA), "configs_Y": cfg_summary(iB)}
            if pn in X3.BENCH_PARTNER and X3.BENCH_PARTNER[pn] in psis:
                Pp = np.hstack([Psi, psis[X3.BENCH_PARTNER[pn]]]); mA2, _ = crossfit_nn(Pp, A, folds, configs); mB2, _ = crossfit_nn(Pp, B, folds, configs)
                bA = max(0.0, 1 - np.mean((A - mA2) ** 2) / np.mean(rA ** 2)); bB = max(0.0, 1 - np.mean((B - mB2) ** 2) / np.mean(rB ** 2))
                rec.update({"bench_partner": X3.BENCH_PARTNER[pn], "bench_r2_X_nn": bA, "bench_r2_Y_nn": bB, "bench_nn": float(np.sqrt(bA * bB))})
            out.append(rec); pd.DataFrame(out).to_csv(OUT / "nn_check_exp3.csv", index=False)
            log(f"exp3 {hyp} psi={pn} (d={Psi.shape[1]}): rho nn {rec['rho_nn']:.3f} (se {rec['se_nn']:.3f}) vs ridge {rec['rho_lin']:.3f} | R2 X nn {rec['R2_X_nn']:.3f} vs {rec['R2_X_lin']:.3f}, "
                f"Y nn {rec['R2_Y_nn']:.3f} vs {rec['R2_Y_lin']:.3f} | val R2 {rec['val_r2_X']:.3f}/{rec['val_r2_Y']:.3f}, iters {rec['iters']:.0f}, configs {rec['configs_X']} / {rec['configs_Y']}"
                + (f" | bench nn {rec['bench_nn']:.3f}" if "bench_nn" in rec else ""))
    curve = []; Psi = psis["E1"]; sizes = [500, 2000] if smoke else X3.N_SUB
    for ns in sizes:
        for sd in range(n_sub):
            rng = np.random.default_rng(sd); idx = rng.choice(n, ns, replace=False); f = crossfit_folds(ns, K, sd)     # as exp3_llm.sub_rep
            mA, _ = crossfit_nn(Psi[idx], A[idx], f, configs, seed=sd); mB, _ = crossfit_nn(Psi[idx], B0[idx], f, configs, seed=sd)
            lA, lB = fit_predict_crossfit(Psi[idx], A[idx], K, sd, "ridge", f), fit_predict_crossfit(Psi[idx], B0[idx], K, sd, "ridge", f)
            curve.append({"n": ns, "seed": sd, "psi": "E1", **tagged(test(A[idx] - mA, B0[idx] - mB), "nn"), **tagged(test(A[idx] - lA, B0[idx] - lB), "lin"),
                          "R2_X_nn": r2(A[idx], mA), "R2_X_lin": r2(A[idx], lA)})
        c = pd.DataFrame(curve); c = c[c.n == ns]; log(f"exp3 curve n={ns}: rho nn median {c.rho_nn.median():.3f} (rej {c.reject_nn.mean():.2f}) vs ridge {c.rho_lin.median():.3f} (rej {c.reject_lin.mean():.2f})")
        pd.DataFrame(curve).to_csv(OUT / "nn_check_exp3_curve.csv", index=False)

# ------------------------------------------------------------------ Experiment 4
def phase_exp4(configs, resample, views_sel, smoke):
    import exp4_truth as T
    from sklearn.decomposition import PCA
    stem = REGEN.stem; n_src = int(re.search(r"_n(\d+)_", stem).group(1))
    keep, M, by = T.load_draws(REGEN); obs = T.load_observed("hf", keep, by, n_src); keep = [i for i in keep if i in obs]; n = len(keep); ck = f"{stem}_{n}"
    src = {r["idx"]: r["Z"] for r in (json.loads(l) for l in T.SOURCE.read_text().splitlines() if l.strip())}; Z = [src[i] for i in keep]; labels = [obs[i]["X_label"] for i in keep]
    E1Z, E2Z = embed_texts(Z, E1_NAME, f"{ck}_Z"), embed_texts(Z, E2_NAME, f"{ck}_Z")
    E1Xo = embed_texts([obs[i]["X"] for i in keep], E1_NAME, f"{ck}_Xobshf"); E1Yo = embed_texts([obs[i]["Y0"] for i in keep], E1_NAME, f"{ck}_Yobshf")
    E1Xd = embed_texts([by[i][m]["X"] for i in keep for m in range(M)], E1_NAME, f"{ck}_Xdraws").reshape(n, M, -1)
    E1Yd = embed_texts([by[i][m]["Y0"] for i in keep for m in range(M)], E1_NAME, f"{ck}_Ydraws").reshape(n, M, -1)
    folds = crossfit_folds(n, K, SEED); dirs = T.fold_directions(E1Xo, labels, folds)
    Xo, Yo = T.apply_directions(dirs, folds, E1Xo), T.apply_directions(dirs, folds, E1Yo); Xm, Ym = T.apply_directions(dirs, folds, E1Xd), T.apply_directions(dirs, folds, E1Yd)
    zc = T.z_components(Xm, Ym); log(f"exp4: {n} posts, M = {M}; post shares {zc['var_EX_Z']/zc['var_X']:.3f} / {zc['var_EY_Z']/zc['var_Y']:.3f}")
    H = np.load(T.LLM / f"hidden_qwen3-1.7b_n{n_src}.npz"); pos = {int(i): j for j, i in enumerate(H["idx"])}; sel = np.array([pos[i] for i in keep])
    views = {"E1": E1Z, "E2": E2Z}
    for name in ("mean_last", "tok_last"):
        Hv = np.asarray(H[name][sel], dtype=float); P = PCA(768, random_state=SEED).fit_transform(Hv)
        for d in (64, 128, 256, 384, 768): views[f"{name}:{d}"] = P[:, :d]
        views[f"{name}:full"] = Hv
    views = {k: views[k] for k in views_sel if k in views}
    rng = np.random.default_rng(SEED); R = resample; ar = np.arange(n); mx_, my_ = rng.integers(M, size=(R, n)), rng.integers(M, size=(R, n))   # the first R null datasets of exp4_truth.py
    Xres, Yres = np.stack([Xm[ar, mx_[r]] for r in range(R)], 1), np.stack([Ym[ar, my_[r]] for r in range(R)], 1)
    rows, rows_d = [], []
    for name, Psi in views.items():
        do_res = R > 0 and name in EXP4_RES_VIEWS
        Tg = np.hstack([Xo[:, None], Yo[:, None], Xm, Ym] + ([Xres, Yres] if do_res else [])); sel = [None, None] + [0] * M + [1] * M + ([0] * R + [1] * R if do_res else [])
        That, info = crossfit_nn(Psi, Tg, folds, configs, select=sel)   # configurations chosen on the observed X and Y, reused for the draws and the null datasets
        mXo, mYo, mX, mY = That[:, 0], That[:, 1], That[:, 2:2 + M], That[:, 2 + M:2 + 2 * M]
        pc = T.psi_components(Xm, Ym, mX, mY); s = T.sensitivity(zc, pc); B, Bt = s["B_star"], abs(s["pred_rho"]); ob = T.dataset_stats(Xo - mXo, Yo - mYo, B, Bt)
        cs = lambda mm, xx: float(np.mean([T.cov(mm[:, m], xx[:, m]) ** 2 / np.var(mm[:, m]) for m in range(M)])); slope = lambda m_, x_: float(T.cov(m_, x_) / np.var(m_))   # slope of X on its prediction: > 1 shrunk, < 1 overdispersed
        cs_X, cs_Y = cs(mX, Xm), cs(mY, Ym)
        dr = [T.dataset_stats(Xm[:, m] - mX[:, m], Ym[:, m] - mY[:, m], B, Bt) for m in range(M)]; rate = lambda L, key: float(np.mean([d[key] for d in L]))
        rec = {"psi": name, "d": Psi.shape[1], "n": n, "M": M, "rho_nn": ob["rho"], "se_nn": ob["se"], "RV_alpha_nn": ob["RV_alpha"], "R2_X_nn": r2(Xo, mXo), "R2_Y_nn": r2(Yo, mYo),
               "val_r2": info["val_r2"], "iters": info["iters"], "configs": cfg_summary(info), **s, "share_psi_X_cs": cs_X / zc["var_X"], "share_psi_Y_cs": cs_Y / zc["var_Y"], "slope_X_nn": slope(mXo, Xo), "slope_Y_nn": slope(mYo, Yo),
               "rho_draws_mean": float(np.mean([d["rho"] for d in dr])), "rej_std_draws": rate(dr, "reject_std"), "rej_B_draws": rate(dr, "reject_B"), "rej_lamB_draws": rate(dr, "reject_lamB"), "rej_Bhalf_draws": rate(dr, "reject_Bhalf")}
        if config_feasible(n, Psi.shape[1], K):     # least squares on the same folds, as in the paper
            L = fit_predict_crossfit(Psi, Tg[:, :2 + 2 * M], K, SEED, "ols", folds); sl = T.sensitivity(zc, T.psi_components(Xm, Ym, L[:, 2:2 + M], L[:, 2 + M:2 + 2 * M])); lo = T.dataset_stats(Xo - L[:, 0], Yo - L[:, 1], sl["B_star"], abs(sl["pred_rho"]))
            rec.update({"rho_lin": lo["rho"], "se_lin": lo["se"], "R2_X_lin": r2(Xo, L[:, 0]), "R2_Y_lin": r2(Yo, L[:, 1]), "B_star_lin": sl["B_star"], "lambda_lin": sl["lambda"], "R2_psi_X_lin": sl["R2_psi_X"], "R2_psi_Y_lin": sl["R2_psi_Y"], "pred_rho_lin": sl["pred_rho"], "R2_view_X_pop_lin": sl["R2_view_X_pop"], "R2_view_Y_pop_lin": sl["R2_view_Y_pop"],
                        "share_psi_X_cs_lin": cs(L[:, 2:2 + M], Xm) / zc["var_X"], "share_psi_Y_cs_lin": cs(L[:, 2 + M:2 + 2 * M], Ym) / zc["var_Y"], "slope_X_lin": slope(L[:, 0], Xo), "slope_Y_lin": slope(L[:, 1], Yo)})
        if do_res:
            mXr, mYr = That[:, 2 + 2 * M:2 + 2 * M + R], That[:, 2 + 2 * M + R:]; rs = [T.dataset_stats(Xres[:, r] - mXr[:, r], Yres[:, r] - mYr[:, r], B, Bt) for r in range(R)]
            rec.update({"R": R, "rho_res_mean": float(np.mean([d["rho"] for d in rs])), "rho_res_sd": float(np.std([d["rho"] for d in rs], ddof=1)), "rej_std_res": rate(rs, "reject_std"), "rej_B_res": rate(rs, "reject_B"), "rej_lamB_res": rate(rs, "reject_lamB"), "rej_Bhalf_res": rate(rs, "reject_Bhalf")})
            rows_d += [{"psi": name, "kind": "resampled", "k": r, **d} for r, d in enumerate(rs)]
        rows_d += [{"psi": name, "kind": "draw", "k": m, **d} for m, d in enumerate(dr)]; rows.append(rec)
        pd.DataFrame(rows).to_csv(OUT / "nn_check_exp4.csv", index=False); pd.DataFrame(rows_d).to_csv(OUT / "nn_check_exp4_datasets.csv", index=False)
        log(f"exp4 {name} (d={Psi.shape[1]}): rho nn {rec['rho_nn']:.3f} (se {rec['se_nn']:.3f})" + (f" vs ols {rec['rho_lin']:.3f}" if "rho_lin" in rec else "") +
            f" | R2 X nn {rec['R2_X_nn']:.3f}" + (f" vs {rec['R2_X_lin']:.3f}" if "R2_X_lin" in rec else "") + f", Y nn {rec['R2_Y_nn']:.3f}" + (f" vs {rec['R2_Y_lin']:.3f}" if "R2_Y_lin" in rec else "") +
            f" | identified: R2_psi X {s['R2_psi_X']:.3f} Y {s['R2_psi_Y']:.3f} lambda {s['lambda']:.3f} B* {B:.3f} pred {s['pred_rho']:.3f} (view shares {s['R2_view_X_pop']:.3f}/{s['R2_view_Y_pop']:.3f}, CS {rec['share_psi_X_cs']:.3f}/{rec['share_psi_Y_cs']:.3f}, slopes {rec['slope_X_nn']:.2f}/{rec['slope_Y_nn']:.2f})" +
            (f" vs ols: R2_psi {rec['R2_psi_X_lin']:.3f}/{rec['R2_psi_Y_lin']:.3f} lambda {rec['lambda_lin']:.3f} B* {rec['B_star_lin']:.3f} (CS {rec['share_psi_X_cs_lin']:.3f}/{rec['share_psi_Y_cs_lin']:.3f}, slopes {rec['slope_X_lin']:.2f}/{rec['slope_Y_lin']:.2f})" if "B_star_lin" in rec else "") +
            f" | draws: reject std {rec['rej_std_draws']:.2f} B* {rec['rej_B_draws']:.2f} lamB* {rec['rej_lamB_draws']:.2f}" + (f" | resampled ({R}): std {rec['rej_std_res']:.2f} B* {rec['rej_B_res']:.2f} lamB* {rec['rej_lamB_res']:.2f} B*/2 {rec['rej_Bhalf_res']:.2f}" if do_res else "") +
            f" | val R2 {info['val_r2']:.3f}, iters {info['iters']:.0f}, configs {rec['configs']}")
    json.dump({"n": n, "M": M, "R": R, "views": list(views), **zc, "configs": [f"{h}/{a}" for h, a in configs], "seeds": SEEDS}, open(OUT / "nn_check_exp4.json", "w"), indent=1)

# ------------------------------------------------------------------ application (parent-comment design)
def phase_app(configs, rater, smoke):
    import app_cmv_deep as D
    from app_cmv import corr_stats
    df = D.load_rows(rater); par = pd.read_parquet(D.PARENTS); V = dict(np.load(D.VIEWS)); log(f"app: rows {len(df)}, ratings {df.rating.notna().sum()}")
    pidx = {p: i for i, p in enumerate(par.parent)}; idx = np.array([pidx[p] for p in df.parent]); strata = df.parent.values; clusters = df.root.values
    lin = pd.read_csv(RESULTS_DIR / f"app_cmv_deep_results_{rater}.csv"); out, ben = [], []
    pairs = D.PAIRS_MAIN[:1] if smoke else D.PAIRS_MAIN
    # resumable: pairs whose rows (and, for the pairs with all embeddings, both benchmarks) are already in the result files are kept and skipped
    fo, fb = OUT / "nn_check_app.csv", OUT / "nn_check_app_bench.csv"
    if fo.exists() and not smoke:
        old, oldb = pd.read_csv(fo), (pd.read_csv(fb) if fb.exists() else pd.DataFrame(columns=["X", "Y"]))
        for xn, yn in pairs:
            need = APP_VIEWS if (xn, yn) in APP_PAIRS_ALL_VIEWS else ["E1"]; have = set(old[(old.X == xn) & (old.Y == yn)].psi)
            nb = len(oldb[(oldb.X == xn) & (oldb.Y == yn)]) if len(oldb) else 0
            if set(need) <= have and (nb == 2 or (xn, yn) not in APP_PAIRS_ALL_VIEWS):
                out += old[(old.X == xn) & (old.Y == yn)].to_dict("records"); ben += oldb[(oldb.X == xn) & (oldb.Y == yn)].to_dict("records") if nb else []; log(f"app {xn}/{yn}: complete in {fo.name}, skipped")
        done = {(r["X"], r["Y"]) for r in out}; pairs = [pr for pr in pairs if pr not in done]
    for xn, yn in pairs:
        X = df[D.XS[xn]].values.astype(float); Y = df[D.YS[yn]].values.astype(float); m = ~np.isnan(X) & ~np.isnan(Y); Xm, Ym, sm, cm, im = X[m], Y[m], strata[m], clusters[m], idx[m]
        keep = pd.Series(sm).groupby(sm).transform("size").values >= 2; Xm, Ym, sm, cm, im = Xm[keep], Ym[keep], sm[keep], cm[keep], im[keep]
        fm = list(GroupKFold(K).split(np.zeros(len(Xm)), groups=cm)); n = len(Xm); st = lin[(lin.X == xn) & (lin.Y == yn)].set_index("psi")
        names = APP_VIEWS if (xn, yn) in APP_PAIRS_ALL_VIEWS and not smoke else ["E1"]
        psis = {"E1": V["E1"][im], "E2": V["E2"][im], "[E1,E2]": np.hstack([V["E1"][im], V["E2"][im]]), "M": V["M"][im], "[E1,M]": np.hstack([V["E1"][im], V["M"][im]]), "L300": V["L300"][im]}
        preds = {}
        for pn in names:
            Psi = psis[pn]; mA, iA = crossfit_nn(Psi, Xm, fm, configs); mB, iB = crossfit_nn(Psi, Ym, fm, configs); preds[pn] = (mA, mB)
            lA, lB = fit_predict_crossfit(Psi, Xm, K, SEED, "ols", fm), fit_predict_crossfit(Psi, Ym, K, SEED, "ols", fm)
            r = corr_stats(Xm - mA, Ym - mB, cm); rl = corr_stats(Xm - lA, Ym - lB, cm)
            rec = {"X": xn, "Y": yn, "psi": pn, "d": Psi.shape[1], "n": n, "rho_nn": r["rho"], "se_nn": r["se"], "RV_alpha_nn": r["RV_alpha"], "R2_X_nn": r2(Xm, mA), "R2_Y_nn": r2(Ym, mB),
                   "rho_lin": rl["rho"], "se_lin": rl["se"], "RV_alpha_lin": rl["RV_alpha"], "R2_X_lin": r2(Xm, lA), "R2_Y_lin": r2(Ym, lB), "rho_strat": st.loc[D.STRAT, "rho"], "se_strat": st.loc[D.STRAT, "se"], "rho_marg": st.loc[D.MARG, "rho"],
                   "val_r2_X": iA["val_r2"], "val_r2_Y": iB["val_r2"], "iters": (iA["iters"] + iB["iters"]) / 2, "configs_X": cfg_summary(iA), "configs_Y": cfg_summary(iB)}
            out.append(rec); pd.DataFrame(out).to_csv(OUT / "nn_check_app.csv", index=False)
            log(f"app {xn}/{yn} psi={pn} (d={Psi.shape[1]}, n={n}): rho nn {rec['rho_nn']:.4f} (se {rec['se_nn']:.4f}) vs ols {rec['rho_lin']:.4f}, baseline {rec['rho_strat']:.4f}, unconditional {rec['rho_marg']:.4f} | "
                f"R2 X nn {rec['R2_X_nn']:.3f} vs {rec['R2_X_lin']:.3f}, Y nn {rec['R2_Y_nn']:.3f} vs {rec['R2_Y_lin']:.3f} | val R2 {rec['val_r2_X']:.3f}/{rec['val_r2_Y']:.3f}, iters {rec['iters']:.0f}, configs {rec['configs_X']} / {rec['configs_Y']}")
        if "E1" in preds:
            mA1, mB1 = preds["E1"]
            for pn in ("E2", "M"):
                if f"[E1,{pn}]" not in preds: continue
                mA2, mB2 = preds[f"[E1,{pn}]"]; bA = max(0.0, 1 - np.mean((Xm - mA2) ** 2) / np.mean((Xm - mA1) ** 2)); bB = max(0.0, 1 - np.mean((Ym - mB2) ** 2) / np.mean((Ym - mB1) ** 2))
                ben.append({"X": xn, "Y": yn, "view": pn, "r2_X_nn": bA, "r2_Y_nn": bB, "rho_bench_nn": float(np.sqrt(bA * bB))}); log(f"  bench nn E1->{pn}: r2X={bA:.4f} r2Y={bB:.4f} rho_bench={np.sqrt(bA*bB):.4f}")
                pd.DataFrame(ben).to_csv(OUT / "nn_check_app_bench.csv", index=False)

# ------------------------------------------------------------------ macros
def phase_macros(write):
    f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"; pct = lambda v: f"{100*v:.0f}\\%"; M = {}
    p3 = OUT / "nn_check_exp3.csv"
    if p3.exists():
        e3 = pd.read_csv(p3); h0 = e3[e3.hyp == "H0"].set_index("psi"); h1 = e3[e3.hyp == "H1"].set_index("psi")
        for pn, key in (("E1", "Eone"), ("E2", "Etwo"), ("E1+E2", "Both")):
            if pn in h0.index: M[f"resNNThreeRho{key}"] = f3(h0.loc[pn, "rho_nn"]); M[f"resNNThreeSe{key}"] = f3(h0.loc[pn, "se_nn"]); M[f"resNNThreeRtwoX{key}"] = pct(max(h0.loc[pn, "R2_X_nn"], 0)); M[f"resNNThreeRtwoY{key}"] = pct(max(h0.loc[pn, "R2_Y_nn"], 0)); M[f"resNNThreeRtwoX{key}Lin"] = pct(max(h0.loc[pn, "R2_X_lin"], 0)); M[f"resNNThreeRtwoY{key}Lin"] = pct(max(h0.loc[pn, "R2_Y_lin"], 0))
            if pn in h1.index: M[f"resNNThreeRho{key}Alt"] = f3(h1.loc[pn, "rho_nn"])
        M["resNNThreeRhoDiffMax"] = f3((e3.rho_nn - e3.rho_lin).abs().max()); M["resNNThreeRhoMin"] = f3(h0.rho_nn.min()); M["resNNThreeRhoMax"] = f3(h0.rho_nn.max())
        if "bench_nn" in e3.columns and "E1" in h0.index and not np.isnan(h0.loc["E1"].get("bench_nn", np.nan)): M["resNNThreeBenchOneTwo"] = f3(h0.loc["E1", "bench_nn"])
        pc = OUT / "nn_check_exp3_curve.csv"
        if pc.exists():
            c = pd.read_csv(pc); W = {500: "Small", 1000: "Mid", 2000: "Large"}
            for ns, g in c.groupby("n"):
                if ns in W: M[f"resNNThreeCurve{W[ns]}"] = f2(g.rho_nn.median()); M[f"resNNThreeCurve{W[ns]}Lin"] = f2(g.rho_lin.median()); M[f"resNNThreeRej{W[ns]}"] = pct(g.reject_nn.mean())
            M["resNNThreeNsub"] = str(int(c.groupby("n").size().min()))
    p4 = OUT / "nn_check_exp4.csv"
    if p4.exists():
        e4 = pd.read_csv(p4).set_index("psi"); W = {"E1": "Eone", "E2": "Etwo", "mean_last:384": "MeanLast", "mean_last:full": "MeanLastFull", "tok_last:384": "TokLast", "tok_last:full": "TokLastFull"}
        for pn, key in W.items():
            if pn not in e4.index: continue
            r = e4.loc[pn]; M[f"resNNFourRho{key}"] = f3(r.rho_nn); M[f"resNNFourSe{key}"] = f3(r.se_nn); M[f"resNNFourRtwoX{key}"] = pct(max(r.R2_X_nn, 0)); M[f"resNNFourRtwoY{key}"] = pct(max(r.R2_Y_nn, 0))
            M[f"resNNFourBstar{key}"] = f3(r.B_star); M[f"resNNFourLambda{key}"] = f2(r["lambda"]); M[f"resNNFourPredRho{key}"] = f3(r.pred_rho); M[f"resNNFourRtwoPsiX{key}"] = pct(r.R2_psi_X); M[f"resNNFourRtwoPsiY{key}"] = pct(r.R2_psi_Y)
            M[f"resNNFourShareX{key}"] = pct(r.R2_view_X_pop); M[f"resNNFourShareY{key}"] = pct(r.R2_view_Y_pop)
            M[f"resNNFourSlopeX{key}"] = f2(r.slope_X_nn); M[f"resNNFourSlopeY{key}"] = f2(r.slope_Y_nn)
            if "rho_lin" in e4.columns and not np.isnan(r.get("rho_lin", np.nan)):
                M[f"resNNFourRho{key}Lin"] = f3(r.rho_lin); M[f"resNNFourBstar{key}Lin"] = f3(r.B_star_lin); M[f"resNNFourLambda{key}Lin"] = f2(r.lambda_lin); M[f"resNNFourShareX{key}Lin"] = pct(r.R2_view_X_pop_lin); M[f"resNNFourShareY{key}Lin"] = pct(r.R2_view_Y_pop_lin)
                M[f"resNNFourRtwoX{key}Lin"] = f"{100*r.R2_X_lin:.0f}\\%"; M[f"resNNFourRtwoY{key}Lin"] = f"{100*r.R2_Y_lin:.0f}\\%"; M[f"resNNFourSlopeX{key}Lin"] = f2(r.slope_X_lin); M[f"resNNFourSlopeY{key}Lin"] = f2(r.slope_Y_lin)
            pd_ = RESULTS_DIR / "exp4_truth.csv"; dd = OUT / "nn_check_exp4_datasets.csv"
            if pd_.exists() and dd.exists():   # robust test on the network residuals with the least-squares tolerance of exp4_truth.py (the paper's B*)
                tr_ = pd.read_csv(pd_).set_index("psi"); ds = pd.read_csv(dd); ds = ds[(ds.psi == pn) & (ds.kind == "resampled")]
                if pn in tr_.index and len(ds):
                    Bl = float(tr_.loc[pn, "B_star"]); M[f"resNNFourRejBls{key}"] = pct(np.mean([abs(rho) > Bl + critical_value(Bl, se, ALPHA) for rho, se in zip(ds.rho, ds.se)])); M[f"resNNFourBstarLs{key}"] = f3(Bl)
            if "rej_B_res" in e4.columns and not np.isnan(r.get("rej_B_res", np.nan)): M[f"resNNFourRejStd{key}"] = pct(r.rej_std_res); M[f"resNNFourRejB{key}"] = pct(r.rej_B_res); M[f"resNNFourRejLamB{key}"] = pct(r.rej_lamB_res); M[f"resNNFourRejBhalf{key}"] = pct(r.rej_Bhalf_res); M["resNNFourR"] = str(int(r.R))
        hv = e4[e4.index.str.contains(":")]; M["resNNFourRhoMinStates"] = f3(hv.rho_nn.min()); M["resNNFourRhoMaxStates"] = f3(hv.rho_nn.max()); M["resNNFourRhoMinAll"] = f3(e4.rho_nn.min()); M["resNNFourRhoMaxAll"] = f3(e4.rho_nn.max())
        M["resNNFourLambdaMin"] = f2(e4["lambda"].min()); M["resNNFourLambdaMax"] = f2(e4["lambda"].max()); M["resNNFourBstarMin"] = f3(e4.B_star.min()); M["resNNFourBstarMax"] = f3(e4.B_star.max())
        M["resNNFourPredDiffMax"] = f3((e4.rho_nn - e4.pred_rho).abs().max())
        if "rho_lin" in e4.columns: d = e4.rho_nn - e4.rho_lin; M["resNNFourRhoDiffMax"] = f3(d.max()); M["resNNFourRhoDiffMin"] = f3(d.min()); M["resNNFourRhoDiffMean"] = f3(d.mean()); M["resNNFourRhoDiffAbsMax"] = f3(d.abs().max())
    pa = OUT / "nn_check_app.csv"
    if pa.exists():
        ea = pd.read_csv(pa); e1 = ea[ea.psi == "E1"].set_index(["X", "Y"])
        for (x, y), key in ((("politeness", "score"), "Pol"), (("log_length", "score"), "Len")):
            if (x, y) in e1.index: r = e1.loc[(x, y)]; M[f"resNNDeep{key}Eone"] = f3(r.rho_nn); M[f"resNNDeep{key}EoneSe"] = f3(r.se_nn); M[f"resNNDeep{key}EoneLin"] = f3(r.rho_lin); M[f"resNNDeep{key}Strat"] = f3(r.rho_strat); M[f"resNNDeep{key}RtwoX"] = f"{100*max(r.R2_X_nn,0):.1f}\\%"; M[f"resNNDeep{key}RtwoY"] = f"{100*max(r.R2_Y_nn,0):.1f}\\%"; M[f"resNNDeep{key}RtwoXLin"] = f"{100*max(r.R2_X_lin,0):.1f}\\%"; M[f"resNNDeep{key}RtwoYLin"] = f"{100*max(r.R2_Y_lin,0):.1f}\\%"
        M["resNNDeepRhoDiffMax"] = f3((ea.rho_nn - ea.rho_lin).abs().max()); M["resNNDeepBiasMax"] = f3((e1.rho_nn - e1.rho_strat).abs().max()); M["resNNDeepBiasMaxLin"] = f3((e1.rho_lin - e1.rho_strat).abs().max())
        M["resNNDeepGapClosedMax"] = pct(max(0.0, 1 - ((e1.rho_nn - e1.rho_strat).abs() / (e1.rho_lin - e1.rho_strat).abs()).min()))
        pb = OUT / "nn_check_app_bench.csv"
        if pb.exists(): M["resNNDeepBenchMax"] = f3(pd.read_csv(pb).rho_bench_nn.max())
    lines = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()); print(lines)
    if write:
        MAC = PAPER_DIR / "results_macros.tex"; s = MAC.read_text(); head = "% ---------- Neural network regressions (nn_check.py) ----------"
        keep = [l for l in s.splitlines() if not re.match(r"\\newcommand\{\\resNN", l) and l.strip() != head]
        MAC.write_text("\n".join(keep).rstrip("\n") + "\n" + head + "\n" + lines + "\n"); print("written to", MAC)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="exp3,exp4,app,macros"); ap.add_argument("--configs", default="full", choices=list(CONFIGS))
    ap.add_argument("--n-sub", type=int, default=20); ap.add_argument("--resample", type=int, default=50); ap.add_argument("--views", default=",".join(EXP4_VIEWS)); ap.add_argument("--rater", default="jev")
    ap.add_argument("--smoke", action="store_true", help="tiny run: small configs, two subsamples, five null datasets, E1 and E2 only"); ap.add_argument("--write", action="store_true")
    a = ap.parse_args(); ph = a.only.split(","); cfg = CONFIGS["small" if a.smoke else a.configs]
    if a.smoke: a.n_sub, a.resample, a.views = 2, 5, "E1,mean_last:64"
    if "exp3" in ph: phase_exp3(cfg, a.n_sub, a.smoke)
    if "exp4" in ph: phase_exp4(cfg, a.resample, a.views.split(","), a.smoke)
    if "app" in ph: phase_app(cfg, a.rater, a.smoke)
    if "macros" in ph: phase_macros(a.write)
    log("done")
