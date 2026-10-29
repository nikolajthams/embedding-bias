"""Cross-fitted residual estimators, the standard GCM test, the robust test and the
encoder benchmark.

Notation: the paper's tested variables X and Y are called A and B in the code;
Psi = psi(Z) is the representation of the conditioning variable; r_A = A - E[A | Psi]
is the paper's r_psi(X); rho = E[r_A r_B] / (sigma_A sigma_B) is the paper's rho_psi;
r2_A, r2_B are the paper's R^2_psi(X), R^2_psi(Y). The insufficiency tolerance B of
the paper is the argument `tol` of critical_value and robust_test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Regressors
# --------------------------------------------------------------------------- #


class KRRMedian(BaseEstimator, RegressorMixin):
    """RBF kernel ridge regression with the median heuristic for the bandwidth.

    The ridge penalty is chosen by 3-fold cross-validation on a random subsample
    of at most `n_select` training points (cheap), after which the final model is
    fitted on the full training fold with one Cholesky solve. Supports (n,) and
    (n, m) targets."""

    def __init__(self, alphas=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0), n_median=1000, n_select=800,
                 cv=3, random_state=0):
        self.alphas = alphas
        self.n_median = n_median
        self.n_select = n_select
        self.cv = cv
        self.random_state = random_state

    @staticmethod
    def _kernel(X1, X2, gamma):
        sq = (X1**2).sum(1)[:, None] + (X2**2).sum(1)[None, :] - 2.0 * X1 @ X2.T
        np.maximum(sq, 0.0, out=sq)
        return np.exp(-gamma * sq)

    @staticmethod
    def _solve(K, y, alpha):
        Kr = K.copy()
        Kr[np.diag_indices_from(Kr)] += alpha
        c, low = cho_factor(Kr, lower=True, check_finite=False)
        return cho_solve((c, low), y, check_finite=False)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=min(self.n_median, n), replace=False)
        d = euclidean_distances(X[idx])
        med = np.median(d[np.triu_indices_from(d, k=1)])
        self.gamma_ = 1.0 / (2.0 * med**2) if med > 0 else 1.0
        # penalty selection on a subsample
        sel = rng.choice(n, size=min(self.n_select, n), replace=False)
        Xs, ys = X[sel], y[sel]
        Ks = self._kernel(Xs, Xs, self.gamma_)
        kf = KFold(self.cv, shuffle=True, random_state=self.random_state)
        errs = np.zeros(len(self.alphas))
        for tr, te in kf.split(Xs):
            Ktr = Ks[np.ix_(tr, tr)]
            Kte = Ks[np.ix_(te, tr)]
            for j, a in enumerate(self.alphas):
                coef = self._solve(Ktr, ys[tr], a)
                errs[j] += np.mean((ys[te] - Kte @ coef) ** 2)
        self.alpha_ = float(self.alphas[int(np.argmin(errs))])
        self.X_train_ = X
        K = self._kernel(X, X, self.gamma_)
        self.coef_ = self._solve(K, y, self.alpha_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._kernel(X, self.X_train_, self.gamma_) @ self.coef_


def make_regressor(name: str = "ridge", **kw):
    """Factory for the regression procedures used to estimate E[. | Psi].
    The name 'hybrid' (least squares or ridge by the size of the training fold) is
    resolved per fold in fit_predict_crossfit, see resolve_regressor."""
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15)))
    if name == "ridge_fixed":
        return make_pipeline(StandardScaler(), Ridge(alpha=kw.get("alpha", 1.0)))
    if name == "ols":
        return make_pipeline(StandardScaler(), Ridge(alpha=1e-8))
    if name == "krr":
        return make_pipeline(StandardScaler(), KRRMedian(random_state=kw.get("random_state", 0)))
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=kw.get("hidden", (64, 64)),
                early_stopping=True,
                max_iter=kw.get("max_iter", 500),
                random_state=kw.get("random_state", 0),
            ),
        )
    raise ValueError(f"unknown regressor {name}")


OLS_MIN_ROWS_PER_COL = 2.5
"""Reporting rule of Experiment 3 (exp4_local.py, exp4_truth.py; 30 Sept 2026): every conditional mean is
estimated by least squares, and an embedding is reported only when every training fold has at least
OLS_MIN_ROWS_PER_COL rows per column of psi(Z); smaller configurations are omitted. Least squares as soon
as it is feasible does not hold the level, because the standard error ignores the estimation error near
n = d (Experiment 2, psi = E1, lambda = 0: rejection rate 0.51 at 400 training rows for 384 columns, 0.22
at 2.1 rows per column, 0.10 at 4.2). Experiments 1 and 2 use the 'hybrid' rule below (7 Oct 2026)."""

HYBRID_MIN_ROWS_PER_COL = 1.0
"""The regressor rule of Experiments 1 and 2 ('hybrid', 7 Oct 2026, at the author's request): least squares
when the training fold has at least HYBRID_MIN_ROWS_PER_COL rows per column of psi(Z), RidgeCV otherwise.
On the grids of both experiments this is n >= d, and every configuration is reported. The raw files of
Experiment 2 store least squares and ridge per draw on the same folds, and the hybrid columns are derived
at plot time (exp2_text.py), so the threshold can be changed without recomputation. History: threshold 2.5
with ridge below it until 30 Sept 2026; least squares only, configurations below 2.5 omitted, until 7 Oct."""


def ols_feasible(n_train: int, d: int) -> bool:
    """Whether least squares is reported for a training fold with n_train rows on d columns (Experiment 3's rule)."""
    return n_train >= OLS_MIN_ROWS_PER_COL * d


def min_train_rows(n: int, K: int = 5) -> int:
    """Rows in the smallest training fold of a K-fold split of n rows."""
    return n - int(np.ceil(n / K))


def config_feasible(n: int, d: int, K: int = 5) -> bool:
    """Whether a configuration with n rows on a d-dimensional conditioning set is reported under Experiment 3's rule."""
    return ols_feasible(min_train_rows(n, K), d)


def hybrid_choice(n_train: int, d: int) -> str:
    """The rule of Experiments 1 and 2: 'ols' when the training fold has at least HYBRID_MIN_ROWS_PER_COL * d rows, else 'ridge'."""
    return "ols" if n_train >= HYBRID_MIN_ROWS_PER_COL * d else "ridge"


def hybrid_choice_n(n: int, d: int, K: int = 5) -> str:
    """The same rule for a configuration with n rows, resolved by its smallest training fold."""
    return hybrid_choice(min_train_rows(n, K), d)


def resolve_regressor(name: str, n_train: int, d: int) -> str:
    """'hybrid' resolves per training fold to least squares or ridge regression by hybrid_choice."""
    if name == "hybrid":
        return hybrid_choice(n_train, d)
    return name


def fit_predict_crossfit(Xmat, y, K: int = 5, seed: int = 0, regressor="ridge", folds=None):
    """K-fold cross-fitted predictions of E[y | Xmat]. `y` may be (n,) or (n, m).
    Returns out-of-fold predictions with the same shape as y."""
    Xmat = np.asarray(Xmat, dtype=float)
    y = np.asarray(y, dtype=float)
    n = Xmat.shape[0]
    if folds is None:
        folds = list(KFold(K, shuffle=True, random_state=seed).split(np.arange(n)))
    yhat = np.empty_like(y)
    for tr, te in folds:
        model = make_regressor(resolve_regressor(regressor, len(tr), Xmat.shape[1])) if isinstance(regressor, str) else regressor
        model.fit(Xmat[tr], y[tr])
        pred = model.predict(Xmat[te])
        yhat[te] = pred.reshape(y[te].shape)
    return yhat


def crossfit_folds(n: int, K: int = 5, seed: int = 0):
    return list(KFold(K, shuffle=True, random_state=seed).split(np.arange(n)))


# --------------------------------------------------------------------------- #
# The residual correlation, its standard error, the tests
# --------------------------------------------------------------------------- #


@dataclass
class GCMResult:
    rho_hat: float
    se: float
    z: float
    pvalue: float
    RV: float
    RV_alpha: float
    reject: bool
    n: int
    alpha: float
    rA: np.ndarray = field(repr=False)
    rB: np.ndarray = field(repr=False)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("rA", "rB")}


def residual_correlation(rA, rB):
    """rho_hat and its influence-function standard error."""
    rA = np.asarray(rA, dtype=float)
    rB = np.asarray(rB, dtype=float)
    n = rA.shape[0]
    sA = np.sqrt(np.mean(rA**2))
    sB = np.sqrt(np.mean(rB**2))
    a, b = rA / sA, rB / sB
    rho = float(np.mean(a * b))
    IF = a * b - 0.5 * rho * (a**2 + b**2)
    se = float(np.std(IF, ddof=1) / np.sqrt(n))
    return rho, se


def critical_value(tol: float, se: float, alpha: float = 0.05) -> float:
    """Exact least-favourable critical value c = c_alpha(B, se) for the insufficiency tolerance
    B = tol: the unique c >= 0 with
        Phibar(c / se) + Phibar((2 B + c) / se) = alpha,
    i.e. the (1 - alpha)-quantile of |B + se N| - B, N ~ N(0, 1). Under the least-favourable
    null |rho| = B the test 'reject iff |rho_hat| > B + c' has level exactly alpha. It equals
    z_{1-alpha/2} se at B = 0 and tends to z_{1-alpha} se for B >> se."""
    if se <= 0:
        return 0.0
    tol = max(0.0, float(tol))
    hi = stats.norm.ppf(1 - alpha / 2) * se

    def h(c):
        return stats.norm.sf(c / se) + stats.norm.sf((2 * tol + c) / se) - alpha

    if h(hi) >= 0:  # numerically only possible for tol == 0
        return float(hi)
    return float(brentq(h, 0.0, hi, xtol=1e-12 * max(se, 1e-12), rtol=1e-10, maxiter=200))


def robust_test(rho_hat: float, se: float, tol: float, alpha: float = 0.05) -> dict:
    """Robust test with insufficiency tolerance B = tol: reject H0 iff |rho_hat| > B + c_alpha(B, se)."""
    c = critical_value(tol, se, alpha)
    return {"reject": bool(abs(rho_hat) > tol + c), "c": c, "tol": float(tol), "alpha": alpha,
            "stat": float((abs(rho_hat) - tol) / se) if se > 0 else np.inf}


def rv_alpha(rho_hat: float, se: float, alpha: float = 0.05) -> float:
    """Robustness value at level alpha: the smallest tolerance B >= 0 at which the robust test
    does not reject, i.e. the root of g(B) = |rho_hat| - B - c_alpha(B, se), which is strictly
    decreasing. Zero if g(0) <= 0."""
    a = abs(float(rho_hat))
    if se <= 0:
        return a

    def g(r0):
        return a - r0 - critical_value(r0, se, alpha)

    if g(0.0) <= 0:
        return 0.0
    return float(brentq(g, 0.0, a, xtol=1e-10, maxiter=200))


def gcm_from_residuals(rA, rB, alpha: float = 0.05) -> GCMResult:
    rho, se = residual_correlation(rA, rB)
    z = rho / se if se > 0 else np.inf
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    q = stats.norm.ppf(1 - alpha / 2)
    return GCMResult(
        rho_hat=rho,
        se=se,
        z=float(z),
        pvalue=float(p),
        RV=abs(rho),
        RV_alpha=rv_alpha(rho, se, alpha),
        reject=bool(abs(z) > q),
        n=len(rA),
        alpha=alpha,
        rA=np.asarray(rA),
        rB=np.asarray(rB),
    )


def gcm_test(A, B, Psi, regressor="ridge", K: int = 5, seed: int = 0, alpha: float = 0.05,
             folds=None) -> GCMResult:
    """The standard (GCM) test: cross-fitted residuals of A and B on Psi, then the
    standardized residual covariance (the GCM of Shah & Peters, 2020)."""
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    Psi = np.asarray(Psi, dtype=float)
    if Psi.ndim == 1:
        Psi = Psi[:, None]
    if folds is None:
        folds = crossfit_folds(len(A), K, seed)
    rA = A - fit_predict_crossfit(Psi, A, K, seed, regressor, folds)
    rB = B - fit_predict_crossfit(Psi, B, K, seed, regressor, folds)
    return gcm_from_residuals(rA, rB, alpha)


def partial_r2_benchmark(y, Psi, PsiPrime, regressor="ridge", K: int = 5, seed: int = 0,
                         folds=None) -> dict:
    """Cross-fitted partial R^2 of PsiPrime beyond Psi for the outcome y:
    (MSE(y | Psi) - MSE(y | Psi, PsiPrime)) / MSE(y | Psi), clipped at 0.
    A population lower bound of R2_y := E[Delta_y^2] / sigma_y^2 whenever PsiPrime
    is a function of Z."""
    y = np.asarray(y, dtype=float).ravel()
    Psi = np.asarray(Psi, dtype=float)
    PsiPrime = np.asarray(PsiPrime, dtype=float)
    if Psi.ndim == 1:
        Psi = Psi[:, None]
    if PsiPrime.ndim == 1:
        PsiPrime = PsiPrime[:, None]
    if folds is None:
        folds = crossfit_folds(len(y), K, seed)
    if regressor == "hybrid":   # both regressions must use the same procedure: resolve the rule by the richer set
        regressor = resolve_regressor("hybrid", len(folds[0][0]), Psi.shape[1] + PsiPrime.shape[1])
    r0 = y - fit_predict_crossfit(Psi, y, K, seed, regressor, folds)
    r1 = y - fit_predict_crossfit(np.hstack([Psi, PsiPrime]), y, K, seed, regressor, folds)
    mse0, mse1 = np.mean(r0**2), np.mean(r1**2)
    return {"r2": float(max(0.0, (mse0 - mse1) / mse0)), "mse0": float(mse0), "mse1": float(mse1),
            "r0": r0, "r1": r1}


def benchmark_pair(A, B, Psi, PsiPrime, regressor="ridge", K: int = 5, seed: int = 0) -> dict:
    folds = crossfit_folds(len(A), K, seed)
    bA = partial_r2_benchmark(A, Psi, PsiPrime, regressor, K, seed, folds)
    bB = partial_r2_benchmark(B, Psi, PsiPrime, regressor, K, seed, folds)
    rA, rB = bA["r0"], bB["r0"]
    res = gcm_from_residuals(rA, rB)
    return {"r2_A": bA["r2"], "r2_B": bB["r2"], "bench": float(np.sqrt(bA["r2"] * bB["r2"])),
            "mse0_A": bA["mse0"], "mse1_A": bA["mse1"], "mse0_B": bB["mse0"], "mse1_B": bB["mse1"], "gcm": res}
