import numpy as np
import pytest
from scipy import stats

from citext.estimators import critical_value, gcm_from_residuals, robust_test, rv_alpha


def test_critical_value_limits():
    se, alpha = 0.03, 0.05
    assert critical_value(0.0, se, alpha) == pytest.approx(stats.norm.ppf(0.975) * se, rel=1e-6)
    assert critical_value(1.0, se, alpha) == pytest.approx(stats.norm.ppf(0.95) * se, rel=1e-4)
    # monotone in the tolerance and with slope < 1 in magnitude
    cs = [critical_value(r, se, alpha) for r in np.linspace(0, 0.2, 21)]
    d = np.diff(cs) / 0.01
    assert np.all(d <= 0) and np.all(np.abs(d) < 1)


@pytest.mark.parametrize("tol,se", [(0.1, 0.03), (0.02, 0.03), (0.0, 0.05), (0.3, 0.01)])
def test_level_is_exact_at_least_favourable_null(tol, se):
    """|rho| = tol with rho_hat ~ N(+-tol, se^2): rejection rate == alpha."""
    alpha = 0.05
    rng = np.random.default_rng(0)
    N = 400_000
    c = critical_value(tol, se, alpha)
    for sign in (1, -1):
        rho_hat = sign * tol + se * rng.normal(size=N)
        rate = np.mean(np.abs(rho_hat) > tol + c)
        assert rate == pytest.approx(alpha, abs=3 * np.sqrt(alpha * (1 - alpha) / N) + 1e-4)
    # strictly inside the null (|rho| < tol) the test is conservative
    rho_hat = 0.5 * tol + se * rng.normal(size=N)
    assert np.mean(np.abs(rho_hat) > tol + c) <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / N)


def test_rv_alpha_is_the_boundary_of_rejection():
    rng = np.random.default_rng(1)
    for _ in range(20):
        rho_hat, se = rng.uniform(-0.3, 0.3), rng.uniform(0.005, 0.05)
        r = rv_alpha(rho_hat, se, 0.05)
        assert r >= 0
        if r > 0:
            assert robust_test(rho_hat, se, r - 1e-6)["reject"] is True
            assert robust_test(rho_hat, se, r + 1e-6)["reject"] is False
        else:
            assert robust_test(rho_hat, se, 0.0)["reject"] is False
    # rv_alpha is below RV and coincides with the two-sided z-test decision at tol = 0
    rA, rB = rng.normal(size=800), rng.normal(size=800)
    res = gcm_from_residuals(rA, rB)
    assert res.RV_alpha <= res.RV + 1e-12
    assert (res.RV_alpha > 0) == res.reject
