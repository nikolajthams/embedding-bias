"""Check of the estimators of exp4_truth.py on the linear-Gaussian model of Experiment 1, where R^2_psi, lambda and B are
analytic: psi keeps the first q of p coordinates; the dropped coefficients set R^2_psi(X) = a^2/(1+a^2), R^2_psi(Y) = b^2/(1+b^2)
and lambda = the cosine between the dropped coefficient vectors. M replicates redraw the noise given Z. The second case adds
300 pure-noise coordinates to psi so that least squares has a large estimation error: the population quantities must still be
recovered (Cov(mhat, X) is not inflated), while B_total of the total-error theorem exceeds B*."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments")); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp4_truth as T  # noqa: E402
from citext.estimators import fit_predict_crossfit, crossfit_folds  # noqa: E402

def case(n, M, p, q, a, b, lam, noise_cols, seed=0):
    rng = np.random.default_rng(seed); Z = rng.standard_normal((n, p))
    u = np.zeros(p - q); u[0] = 1.0; v = np.zeros(p - q); v[0] = lam; v[1] = np.sqrt(1 - lam**2)
    thX = np.r_[np.ones(q), a * u]; thY = np.r_[np.ones(q), b * v]
    Xm = (Z @ thX)[:, None] + rng.standard_normal((n, M)); Ym = (Z @ thY)[:, None] + rng.standard_normal((n, M))
    Psi = np.hstack([Z[:, :q], rng.standard_normal((n, noise_cols))]) if noise_cols else Z[:, :q]
    folds = crossfit_folds(n, 5, seed); That = fit_predict_crossfit(Psi, np.hstack([Xm, Ym]), 5, seed, "ols", folds)
    s = T.sensitivity(T.z_components(Xm, Ym), T.psi_components(Xm, Ym, That[:, :M], That[:, M:]))
    rho = np.mean([np.corrcoef(Xm[:, m] - That[:, m], Ym[:, m] - That[:, M + m])[0, 1] for m in range(M)])
    truth = {"R2_psi_X": a**2 / (1 + a**2), "R2_psi_Y": b**2 / (1 + b**2), "lambda": lam}; truth["B_star"] = np.sqrt(truth["R2_psi_X"] * truth["R2_psi_Y"])
    print(f"n={n} M={M} d={Psi.shape[1]}: " + ", ".join(f"{k} {s[k]:.3f} (true {truth[k]:.3f})" for k in truth) + f"; pred rho {s['pred_rho']:.3f} vs mean rho_hat {rho:.3f}; B_total {s['B_total']:.3f}; mse_est/sigma2 {s['mse_est_X']/s['sigma2_psi_X']:.3f}")
    for k in truth: assert abs(s[k] - truth[k]) < 0.04, (k, s[k], truth[k])
    assert abs(s["pred_rho"] - rho) < 0.03
    return s

def test_identification_low_dim():
    case(n=3000, M=4, p=10, q=3, a=1.0, b=0.8, lam=0.7, noise_cols=0)

def test_identification_large_estimation_error():
    s = case(n=3000, M=4, p=10, q=3, a=1.0, b=0.8, lam=0.7, noise_cols=300)   # d = 303 on 2400 training rows
    assert s["B_total"] > s["B_star"] + 0.03                                  # the total-error bound is inflated, B* is not

def test_identification_two_draws_negative_alignment():
    case(n=3000, M=2, p=10, q=3, a=0.6, b=1.5, lam=-0.4, noise_cols=300)

if __name__ == "__main__":
    test_identification_low_dim(); test_identification_large_estimation_error(); test_identification_two_draws_negative_alignment(); print("OK")
