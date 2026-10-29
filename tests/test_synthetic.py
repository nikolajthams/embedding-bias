import numpy as np
import pytest

from citext.estimators import gcm_from_residuals, gcm_test, robust_test
from citext.synthetic import (LinearConfig, conditional_means_given_psi, population,
                              partial_r2_population, sample)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_population_formulas_by_monte_carlo(nonlinear, lam):
    cfg = LinearConfig(q=5, r2_A=0.2, r2_B=0.3, lam=lam, nonlinear=nonlinear)
    pop = population(cfg)
    assert pop["rho"] == pytest.approx(lam * np.sqrt(0.2 * 0.3), abs=1e-12)
    assert pop["r2_A"] == pytest.approx(0.2, abs=1e-12)
    rng = np.random.default_rng(1)
    d = sample(cfg, 400_000, rng)
    mA, mB = conditional_means_given_psi(cfg, d["Z"])
    rA, rB = d["A"] - mA, d["B"] - mB
    rho_mc = np.mean(rA * rB) / np.sqrt(np.mean(rA**2) * np.mean(rB**2))
    r2A_mc = np.var(d["fA"] - mA) / np.mean(rA**2)
    assert rho_mc == pytest.approx(pop["rho"], abs=0.006)
    assert r2A_mc == pytest.approx(pop["r2_A"], abs=0.006)


def test_h1_population_rho():
    cfg = LinearConfig(q=20, beta=0.3)
    pop = population(cfg)
    assert pop["rho"] == pytest.approx(0.3 / np.sqrt(1 + 0.09), abs=1e-12)
    assert pop["bound"] == 0.0


def test_partial_r2_population_is_lower_bound():
    cfg = LinearConfig(q=5, r2_A=0.1, r2_B=0.1, lam=1.0)
    pop = population(cfg)
    bench = partial_r2_population(cfg, q_prime=10)
    assert 0 <= bench["r2_A"] <= pop["r2_A"]
    assert bench["bench"] <= pop["bound"]
    full = partial_r2_population(cfg, q_prime=20)
    assert full["r2_A"] == pytest.approx(pop["r2_A"], abs=1e-12)


def test_naive_test_has_level_with_sufficient_psi():
    cfg = LinearConfig(q=20)
    rng = np.random.default_rng(2)
    rej = []
    for _ in range(300):
        d = sample(cfg, 300, rng)
        rej.append(gcm_test(d["A"], d["B"], d["Psi"], regressor="ridge", seed=0).reject)
    rate = np.mean(rej)
    assert rate < 0.05 + 3 * np.sqrt(0.05 * 0.95 / 300)


def test_robust_test_and_rv_alpha_consistent():
    rng = np.random.default_rng(3)
    rA, rB = rng.normal(size=500), rng.normal(size=500)
    res = gcm_from_residuals(rA, rB)
    assert robust_test(res.rho_hat, res.se, res.RV_alpha + 1e-9)["reject"] is False
    if res.RV_alpha > 0:
        assert robust_test(res.rho_hat, res.se, res.RV_alpha - 1e-9)["reject"] is True
