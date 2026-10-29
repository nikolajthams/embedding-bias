"""Experiment 4, power: alternatives built from the replicates of exp4_regen.py by coupling the comment and the summary draws
within each post, and the rejection rates of the tests against them.

For post i with pool draws X_{i1..iP} and Y_{i1..iP} (conditionally i.i.d. given the post and independent of each other, which
tau_id = 0 in exp4_truth.py confirms), a dataset takes m_X uniform and, with probability pi, the summary draw with the same
(sign +) or the opposite (sign -) sentiment rank as X_{i m_X} among the pool; otherwise an independent uniform draw. The laws of
X given Z and of Y given Z are those of the null, so R^2_psi(X), R^2_psi(Y), lambda, the true insufficiency B* and the true bias
|lambda| B* are unchanged for every embedding, and the conditional correlation given the post is exactly
    rho_id = pi * mean_i C_i^{+-} / sqrt(mean_i Var(X_i.) * mean_i Var(Y_i.)),
with C_i^+ the covariance of the sorted comment and summary draws (comonotone) and C_i^- with one order reversed; pi = 0 is H_0.
By the decomposition of the theorem "The pair", rho_psi = sqrt((1 - R^2_psi(X))(1 - R^2_psi(Y))) rho_id + lambda B* under every
alternative, which gives the predicted rejection probabilities reported next to the realized rates.
The pool is all M draws, so pi = 0 is the mechanism of the null datasets of exp4_truth.py. An oracle conditions on the post
itself through the replicates: the residual correlation test of X - mean(unused comment draws) and Y - mean(unused summary draws),
the means taken over the M - 2 draws of the post that the dataset does not use. Under H_0 these means are independent of the
dataset given the post, so the oracle has exact level; its residual variances carry the error of the means (order 1/(M - 2)) and
the rank coupling ties the selected summary draw weakly to the other comment draws, so its power approximates that of the test
with the post observed. The slope of rho_psi in rho_id is computed with the within-post variances of the pool (ddof 0, the law of
the constructed alternative), which is what `factor` reports; the theorem's population factor is sqrt((1-R^2_psi(X))(1-R^2_psi(Y))).
Outputs: results/exp4_power.csv (one row per embedding and target rho_id), exp4_power.json; --write adds the \\resFourPower* block
to paper/results_macros.tex. The figure (right panel of exp4_rejection.pdf) is made by fig_exp4.py.
usage: python experiments/exp4_power.py ~/.cache/ci_text/llm/regen_qwen3-1.7b_n3000_M20.jsonl --observed hf --write
       [--views E1,E2,mean_last:384] [--grid 0:0.8:0.025] [--resample 200] [--truth results/exp4_truth.csv] [--regressor ols|mlp]
Regressor (7 Oct 2026): the paper uses least squares. --regressor mlp fits the small networks of citext/nn.py with the configurations that
exp4_truth.py chose on the observed data (exp4_truth_nn.json) and writes exp4_power_nn.csv and exp4_power_nn.json (resumable per embedding
and target); its --write adds a separate block of \\resFourPowerNn* macros (not used in the paper) with the least squares values as ...Ls.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import exp4_truth as T
from citext.estimators import fit_predict_crossfit, crossfit_folds, residual_correlation, critical_value
from citext.nn import crossfit_nn
from citext.paths import RESULTS_DIR, PAPER_DIR
from citext.text_semi import E1_NAME, E2_NAME, embed_texts
ALPHA, SEED, K = T.ALPHA, T.SEED, T.K
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ the alternatives (pure numpy; checked by test_exp4_power.py)
def pool_stats(Xp, Yp):
    """Attainable conditional correlations of the pool: comonotone (rho_max > 0) and countermonotone (rho_min < 0) couplings."""
    xs, ys = np.sort(Xp, 1), np.sort(Yp, 1); xb, yb = Xp.mean(1), Yp.mean(1)
    Cp, Cm = (xs * ys).mean(1) - xb * yb, (xs * ys[:, ::-1]).mean(1) - xb * yb; vx, vy = Xp.var(1).mean(), Yp.var(1).mean()
    return {"P": Xp.shape[1], "var_within_X": float(vx), "var_within_Y": float(vy), "C_plus": float(Cp.mean()), "C_minus": float(Cm.mean()),
            "rho_max": float(Cp.mean() / np.sqrt(vx * vy)), "rho_min": float(Cm.mean() / np.sqrt(vx * vy))}

def weight(target, ps):
    """Mixing weight pi and sign that give the conditional correlation `target`; None if unattainable."""
    if target == 0: return 0.0, 1
    pi = target / ps["rho_max"] if target > 0 else target / ps["rho_min"]
    return (pi, 1 if target > 0 else -1) if pi <= 1 + 1e-9 else (None, None)

class Coupler:
    def __init__(self, Xp, Yp):
        self.Xp, self.Yp = Xp, Yp; self.n, self.P = Xp.shape; self.ar = np.arange(self.n)
        self.rank_X = np.argsort(np.argsort(Xp, 1), 1)      # rank of every comment draw within its post
        self.order_Y = np.argsort(Yp, 1)                   # order_Y[i, r]: the summary draw of rank r in post i
        self.sum_X, self.sum_Y = Xp.sum(1), Yp.sum(1)
    def draw(self, pi, sign, rng):
        """One dataset (X_i, Y_i)_i with conditional correlation pi * rho_max (sign +) or pi * rho_min (sign -), and the
        oracle's post means over the draws the dataset does not use."""
        mX = rng.integers(self.P, size=self.n); r = self.rank_X[self.ar, mX]
        mY_c = self.order_Y[self.ar, r if sign > 0 else self.P - 1 - r]; mY_u = rng.integers(self.P, size=self.n)
        mY = np.where(rng.random(self.n) < pi, mY_c, mY_u); X, Y = self.Xp[self.ar, mX], self.Yp[self.ar, mY]
        same = mX == mY; k = self.P - 2 + same
        XH = (self.sum_X - X - np.where(same, 0.0, self.Xp[self.ar, mY])) / k; YH = (self.sum_Y - Y - np.where(same, 0.0, self.Yp[self.ar, mX])) / k
        return X, Y, XH, YH

def predicted_rate(mu, se, B):
    """P(|N(mu, se^2)| > B + c_alpha(B, se)): the normal approximation of the rejection probability at rho_psi = mu."""
    c = critical_value(B, se, ALPHA); return float(norm.sf((B + c - mu) / se) + norm.cdf((-B - c - mu) / se))

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("regen"); ap.add_argument("--observed", choices=("0", "hf"), default="hf"); ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--views", default="E1,E2,mean_last:384"); ap.add_argument("--grid", default="0:0.8:0.025", help="target rho_id values lo:hi:step; negative values give the countermonotone coupling"); ap.add_argument("--resample", type=int, default=200)
    ap.add_argument("--truth", default=None, help="B* and lambda per embedding, from exp4_truth.py (default results/exp4_truth.csv, or exp4_truth_nn.csv with --regressor mlp)"); ap.add_argument("--out", default=str(RESULTS_DIR)); ap.add_argument("--write", action="store_true")
    ap.add_argument("--regressor", choices=("ols", "mlp"), default="ols", help="least squares (files without suffix; the comparison) or the networks of citext.nn with the configurations chosen in exp4_truth.py (files exp4_power_nn*; the paper's)")
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); sfx = "" if a.regressor == "ols" else "_nn"
    truth_path = Path(a.truth).expanduser() if a.truth else RESULTS_DIR / f"exp4_truth{sfx}.csv"; chosen = json.load(open(truth_path.with_suffix(".json")))["chosen"] if a.regressor == "mlp" else None
    regen = Path(a.regen).expanduser(); stem = regen.stem; n_src = int(re.search(r"_n(\d+)_", stem).group(1))
    keep, M, by = T.load_draws(regen, a.M); obs = T.load_observed(a.observed, keep, by, n_src); keep = [i for i in keep if i in obs]; n = len(keep); ck = f"{stem}_{n}"
    src = {r["idx"]: r["Z"] for r in (json.loads(l) for l in T.SOURCE.read_text().splitlines() if l.strip())}; Z = [src[i] for i in keep]
    E1Z, E2Z = embed_texts(Z, E1_NAME, f"{ck}_Z"), embed_texts(Z, E2_NAME, f"{ck}_Z")
    E1Xo = embed_texts([obs[i]["X"] for i in keep], E1_NAME, f"{ck}_Xobs{a.observed}")
    E1Xd = embed_texts([by[i][m]["X"] for i in keep for m in range(M)], E1_NAME, f"{ck}_Xdraws").reshape(n, M, -1)
    E1Yd = embed_texts([by[i][m]["Y0"] for i in keep for m in range(M)], E1_NAME, f"{ck}_Ydraws").reshape(n, M, -1)
    folds = crossfit_folds(n, K, SEED); dirs = T.fold_directions(E1Xo, [obs[i]["X_label"] for i in keep], folds)
    Xm, Ym = T.apply_directions(dirs, folds, E1Xd), T.apply_directions(dirs, folds, E1Yd)
    P = M; ps = pool_stats(Xm, Ym); cp = Coupler(Xm, Ym); log(f"{n} posts, pool of {P} draws: attainable rho_id in [{ps['rho_min']:.3f}, {ps['rho_max']:.3f}]")
    lo, hi, step = (float(v) for v in a.grid.split(":")); targets = [round(t, 4) for t in np.arange(lo, hi + step / 2, step)]
    targets = [t for t in targets if weight(t, ps)[0] is not None]
    truth = pd.read_csv(truth_path).set_index("psi"); views = T.build_views(keep, E1Z, E2Z, T.LLM / f"hidden_qwen3-1.7b_n{n_src}.npz", [v.split(":")[0] for v in a.views.split(",")], sorted({int(v.split(":")[1]) for v in a.views.split(",") if ":" in v}) or [384], True)
    views = {k: v for k, v in views.items() if k in a.views.split(",")}
    rng = np.random.default_rng(SEED + 7); R = a.resample
    data = {t: [cp.draw(*weight(t, ps), rng) for _ in range(R)] for t in targets}                     # the same datasets for every embedding
    orc = {t: [residual_correlation(X - XH, Y - YH) for X, Y, XH, YH in data[t]] for t in targets}
    fo = out / f"exp4_power{sfx}.csv"; rows = pd.read_csv(fo).to_dict("records") if fo.exists() else []   # resumable per embedding and target
    done = {(r["psi"], round(float(r["rho_id"]), 4)) for r in rows}
    if rows: log(f"{len(rows)} rows already in {fo.name}, their (embedding, target) pairs are skipped")
    for name, Psi in views.items():
        tr = truth.loc[name]; B, lam = float(tr.B_star), float(tr["lambda"]); Bt = abs(lam) * B
        fac = float(np.sqrt(ps["var_within_X"] * ps["var_within_Y"] / (tr.sigma2_psi_X * tr.sigma2_psi_Y))); fac_pop = float(np.sqrt((1 - tr.R2_psi_X) * (1 - tr.R2_psi_Y)))
        for t in targets:
            if (name, round(t, 4)) in done: continue
            pi, sign = weight(t, ps); Xs = np.stack([d[0] for d in data[t]], 1); Ys = np.stack([d[1] for d in data[t]], 1)
            if a.regressor == "ols": That = fit_predict_crossfit(Psi, np.hstack([Xs, Ys]), K, SEED, "ols", folds)
            else: cf = chosen[name]; That, _ = crossfit_nn(Psi, np.hstack([Xs, Ys]), folds, fixed={**{j: cf["X"] for j in range(R)}, **{R + j: cf["Y"] for j in range(R)}})   # the configurations chosen on the observed data
            rX, rY = Xs - That[:, :R], Ys - That[:, R:]
            st = [residual_correlation(rX[:, r], rY[:, r]) for r in range(R)]; rho = np.array([s[0] for s in st]); se = np.array([s[1] for s in st])
            rej = lambda Btol: float(np.mean([abs(r_) > Btol + critical_value(Btol, s_, ALPHA) for r_, s_ in zip(rho, se)]))
            orho, ose = np.array([o[0] for o in orc[t]]), np.array([o[1] for o in orc[t]]); mu = fac * t + lam * B; sem = float(se.mean())
            rows.append({"psi": name, "d": Psi.shape[1], "rho_id": t, "pi": pi, "sign": sign, "n": n, "P": P, "R": R, "regressor": a.regressor, "B_star": B, "lambda": lam, "factor": fac, "factor_pop": fac_pop, "rho_psi_pred": mu,
                         "rho_mean": float(rho.mean()), "rho_sd": float(rho.std(ddof=1)), "se_mean": sem, "rej_std": rej(0.0), "rej_Bhalf": rej(B / 2), "rej_lamB": rej(Bt), "rej_B": rej(B),
                         "pred_std": predicted_rate(mu, sem, 0.0), "pred_Bhalf": predicted_rate(mu, sem, B / 2), "pred_lamB": predicted_rate(mu, sem, Bt), "pred_B": predicted_rate(mu, sem, B),
                         "oracle_rho_mean": float(orho.mean()), "oracle_se_mean": float(ose.mean()), "rej_oracle": float(np.mean([abs(r_) > critical_value(0.0, s_, ALPHA) for r_, s_ in zip(orho, ose)]))})
            r_ = rows[-1]; pd.DataFrame(rows).to_csv(fo, index=False); log(f"{name} rho_id {t:+.2f} (pi {pi:.2f}): rho {r_['rho_mean']:+.3f} (pred {mu:+.3f}, se {sem:.3f}) | reject std {r_['rej_std']:.2f} B*/2 {r_['rej_Bhalf']:.2f} lamB* {r_['rej_lamB']:.2f} B* {r_['rej_B']:.2f} oracle {r_['rej_oracle']:.2f} (oracle rho {r_['oracle_rho_mean']:+.3f})")
    D = pd.DataFrame(rows); D["_o"] = D.psi.map({v: i for i, v in enumerate(views)}); D = D.sort_values(["_o", "rho_id"]).drop(columns="_o").reset_index(drop=True); D.to_csv(fo, index=False)
    summ = {"regen": str(regen), "observed": a.observed, "regressor": a.regressor, "truth": str(truth_path), "n": n, "M": M, "P": P, "R": R, "grid": a.grid, "views": list(views), **ps}
    json.dump(summ, open(out / f"exp4_power{sfx}.json", "w"), indent=1); log(f"written to {out} (files exp4_power{sfx}*)"); macros(D, summ, write=a.write, regressor=a.regressor, out=out)

# ------------------------------------------------------------------ macros
AT = ((0.05, "Five"), (0.1, "Ten"), (0.15, "Fifteen"), (0.2, "Twenty"), (0.25, "TwentyFive"), (0.3, "Thirty"), (0.35, "ThirtyFive"), (0.4, "Forty"), (0.5, "Fifty"))
def macros(D, summ, write, regressor="ols", out=None):
    """The \\resFourPower* block; with regressor 'mlp' (the paper's) the least squares values of exp4_power.csv are added with the suffix Ls."""
    f2 = lambda v: f"{v:.2f}"; pct = lambda v: f"{100*v:.0f}\\%"
    M = {"resFourPowerP": str(summ["P"]), "resFourPowerR": str(summ["R"]), "resFourPowerRange": f2(summ["rho_max"]), "resFourPowerRangeNeg": f2(-summ["rho_min"]), "resFourPowerRegressor": {"ols": "least squares", "mlp": "small neural networks"}[regressor]}
    def half(d, col): return f2(float(np.interp(0.5, d[col][d.rho_id >= 0].values, d.rho_id[d.rho_id >= 0].values)))   # rho_id at which the rate crosses one half (rates increase on the positive side)
    for key, psi in (("Eone", "E1"), ("Etwo", "E2"), ("MeanLast", "mean_last:384")):
        d = D[D.psi == psi].reset_index(drop=True)
        if not len(d): continue
        r = d.iloc[0]; B, lam, fac = r.B_star, r["lambda"], r.factor
        M[f"resFourPowerFactor{key}"] = f2(fac); M[f"resFourPowerBiasId{key}"] = f2(abs(lam) * B / fac)                 # the bias, in units of rho_id
        M[f"resFourPowerSampleFactor{key}"] = f"{1 / r.factor_pop**2:.0f}"                                             # observations needed for the power of the test with Z observed, theorem "The pair"
        M[f"resFourPowerThreshB{key}"] = f2((B - abs(lam) * B) / fac)                                                   # rho_id above which the test at B* detects
        M[f"resFourPowerThreshNegLamB{key}"] = f2(-2 * abs(lam) * B / fac)                                             # rho_id below which the test at |lambda| B* detects
        M[f"resFourPowerStdMin{key}"] = pct(d.rej_std.min()); M[f"resFourPowerStdMinAt{key}"] = f2(d.rho_id[d.rej_std.idxmin()]); M[f"resFourPowerBhalfMin{key}"] = pct(d[d.rho_id >= 0].rej_Bhalf.min())
        neg, pos = d[d.rho_id < 0], d[d.rho_id > 0]
        if len(neg): M[f"resFourPowerLamBNegMax{key}"] = pct(neg.rej_lamB.max()); M[f"resFourPowerBNegMax{key}"] = pct(neg.rej_B.max()); M[f"resFourPowerOracleNegMin{key}"] = pct(neg[neg.rho_id <= -0.1].rej_oracle.min())
        M[f"resFourPowerLamBMinusOracleMax{key}"] = pct((pos.rej_oracle - pos.rej_lamB).max())                         # largest shortfall of the test at the true bias behind the oracle, positive side
        M[f"resFourPowerHalfLamB{key}"] = half(d, "rej_lamB"); M[f"resFourPowerHalfB{key}"] = half(d, "rej_B"); M[f"resFourPowerHalfOracle{key}"] = half(d, "rej_oracle")
        M[f"resFourPowerLamBAtZero{key}"] = pct(float(d[np.isclose(d.rho_id, 0)].rej_lamB.iloc[0])) if np.isclose(d.rho_id, 0).any() else "TBD"
        for t, w in AT:
            dd = d[np.isclose(d.rho_id, t)]
            if len(dd): rr = dd.iloc[0]; M[f"resFourPowerLamBAt{w}{key}"] = pct(rr.rej_lamB); M[f"resFourPowerBAt{w}{key}"] = pct(rr.rej_B); M[f"resFourPowerOracleAt{w}{key}"] = pct(rr.rej_oracle); M[f"resFourPowerStdAt{w}{key}"] = pct(rr.rej_std)
        M[f"resFourPowerPredDiffMax{key}"] = f2(max((d.rej_lamB - d.pred_lamB).abs().max(), (d.rej_B - d.pred_B).abs().max(), (d.rej_std - d.pred_std).abs().max()))
        M[f"resFourPowerRhoDiffMax{key}"] = f2((d.rho_mean - d.rho_psi_pred).abs().max())                                   # realized mean of rho_hat against the line of the theorem
    if regressor == "mlp" and out is not None and (out / "exp4_power.csv").exists():   # least squares (the comparison, suffix Ls)
        L = pd.read_csv(out / "exp4_power.csv")
        for key, psi in (("Eone", "E1"), ("Etwo", "E2"), ("MeanLast", "mean_last:384")):
            d = L[L.psi == psi].reset_index(drop=True)
            if not len(d): continue
            pos = d[d.rho_id > 0]; M[f"resFourPowerHalfLamB{key}Ls"] = half(d, "rej_lamB"); M[f"resFourPowerHalfB{key}Ls"] = half(d, "rej_B"); M[f"resFourPowerHalfOracle{key}Ls"] = half(d, "rej_oracle")
            M[f"resFourPowerLamBMinusOracleMax{key}Ls"] = pct((pos.rej_oracle - pos.rej_lamB).max()); M[f"resFourPowerBhalfMin{key}Ls"] = pct(d[d.rho_id >= 0].rej_Bhalf.min()); M[f"resFourPowerFactor{key}Ls"] = f2(d.iloc[0].factor)
            M[f"resFourPowerRhoDiffMax{key}Ls"] = f2((d.rho_mean - d.rho_psi_pred).abs().max())
            for t, w in AT:
                dd = d[np.isclose(d.rho_id, t)]
                if len(dd): rr = dd.iloc[0]; M[f"resFourPowerLamBAt{w}{key}Ls"] = pct(rr.rej_lamB); M[f"resFourPowerBAt{w}{key}Ls"] = pct(rr.rej_B); M[f"resFourPowerStdAt{w}{key}Ls"] = pct(rr.rej_std)
    lines = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()); print(lines)
    if write:
        MAC = PAPER_DIR / "results_macros.tex"; s = MAC.read_text(); base = "% ---------- Experiment 4, power against coupled alternatives"
        if regressor == "mlp":   # the network comparison is not in the paper: its block carries the infix Nn
            lines = "\n".join(f"\\newcommand{{\\{k.replace('resFourPower', 'resFourPowerNn', 1)}}}{{{v}}}" for k, v in M.items()); head = base + ", NETWORK COMPARISON (exp4_power.py --regressor mlp; not used in the paper) ----------"
            drop = lambda l: re.match(r"\\newcommand\{\\resFourPowerNn", l) is not None or l.startswith(head)
        else:
            head = base + " ----------"; drop = lambda l: re.match(r"\\newcommand\{\\resFourPower(?!Nn)", l) is not None or (l.startswith(base) and "NETWORK" not in l)
        keep = [l for l in s.splitlines() if not drop(l)]
        MAC.write_text("\n".join(keep).rstrip("\n") + "\n" + head + "\n" + lines + "\n"); print("written to", MAC)

if __name__ == "__main__":
    main()
