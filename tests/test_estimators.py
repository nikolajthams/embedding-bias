import numpy as np
import pytest

from citext.estimators import KRRMedian, benchmark_pair, gcm_test, residual_correlation
from citext.synthetic import LinearConfig, partial_r2_population, population, sample


def test_residual_correlation_se_matches_gcm_under_independence():
    rng = np.random.default_rng(0)
    rA, rB = rng.normal(size=5000), rng.normal(size=5000)
    rho, se = residual_correlation(rA, rB)
    # under independence the IF reduces to a*b, whose sd is ~1, so se ~ 1/sqrt(n)
    assert se == pytest.approx(1 / np.sqrt(5000), rel=0.1)
    assert abs(rho) < 4 * se


def test_benchmark_is_consistent_for_linear_model():
    cfg = LinearConfig(q=5, r2_A=0.2, r2_B=0.2, lam=1.0)
    rng = np.random.default_rng(5)
    d = sample(cfg, 20000, rng)
    b = benchmark_pair(d["A"], d["B"], d["Psi"], d["Z"][:, 5:10], regressor="ridge", seed=0)
    pop_b = partial_r2_population(cfg, 10)
    assert b["r2_A"] == pytest.approx(pop_b["r2_A"], abs=0.02)
    assert b["bench"] <= population(cfg)["bound"] + 0.02
    # the full complement recovers R2 itself
    b_full = benchmark_pair(d["A"], d["B"], d["Psi"], d["Z"][:, 5:], regressor="ridge", seed=0)
    assert b_full["r2_A"] == pytest.approx(population(cfg)["r2_A"], abs=0.02)


def test_krr_recovers_smooth_function():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 3))
    f = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    y = f + 0.3 * rng.normal(size=1500)
    m = KRRMedian().fit(X[:1000], y[:1000])
    mse = np.mean((m.predict(X[1000:]) - f[1000:]) ** 2)
    assert mse < 0.05


def test_gcm_test_returns_valid_pvalue_shape():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(400, 4))
    A = Z[:, 0] + rng.normal(size=400)
    B = Z[:, 1] + rng.normal(size=400)
    r = gcm_test(A, B, Z, regressor="krr")
    assert 0 <= r.pvalue <= 1 and r.RV_alpha <= r.RV
