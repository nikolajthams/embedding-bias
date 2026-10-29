"""The coupled alternatives of exp4_power.py: the mixing weight sets the conditional correlation given the post exactly, and the
laws of X given Z and of Y given Z are those of the null (the draws are only re-paired)."""
import sys; from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from exp4_power import pool_stats, weight, Coupler

def _pool(n=400, P=10, seed=0):
    rng = np.random.default_rng(seed); mu = rng.normal(size=(n, 2))
    return mu[:, [0]] + rng.normal(size=(n, P)), 0.5 * mu[:, [1]] + 0.3 * rng.normal(size=(n, P))

def test_conditional_correlation_matches_target():
    Xp, Yp = _pool(); ps = pool_stats(Xp, Yp); cp = Coupler(Xp, Yp); rng = np.random.default_rng(1)
    assert 0.6 < ps["rho_max"] < 1 and -1 < ps["rho_min"] < -0.6
    for target in (0.0, 0.3, -0.5, ps["rho_max"]):
        pi, sign = weight(target, ps); xb, yb = Xp.mean(1), Yp.mean(1); prods = []
        for _ in range(400):
            X, Y, XH, YH = cp.draw(pi, sign, rng); prods.append(np.mean((X - xb) * (Y - yb)))      # within-post cross products
        rho = np.mean(prods) / np.sqrt(ps["var_within_X"] * ps["var_within_Y"])
        assert abs(rho - target) < 0.02, (target, rho)

def test_marginals_unchanged():
    Xp, Yp = _pool(); ps = pool_stats(Xp, Yp); cp = Coupler(Xp, Yp); rng = np.random.default_rng(2); pi, sign = weight(0.5, ps)
    X, Y, XH, YH = zip(*[cp.draw(pi, sign, rng) for _ in range(300)]); X, Y, XH, YH = (np.stack(v, 1) for v in (X, Y, XH, YH))
    assert abs(X.mean() - Xp.mean()) < 0.01 and abs(Y.mean() - Yp.mean()) < 0.005
    assert abs(X.var(1).mean() - Xp.var(1).mean()) < 0.02 and abs(Y.var(1).mean() - Yp.var(1).mean()) < 0.005   # within-post variances
    assert weight(1.5, ps) == (None, None)

def test_oracle_means_exclude_the_selected_draws():
    Xp, Yp = _pool(n=50, P=6); ps = pool_stats(Xp, Yp); cp = Coupler(Xp, Yp); rng = np.random.default_rng(3); pi, sign = weight(0.4, ps)
    X, Y, XH, YH = cp.draw(pi, sign, rng)
    for i in range(50):
        mX = int(np.where(Xp[i] == X[i])[0][0]); mY = int(np.where(Yp[i] == Y[i])[0][0]); rest = [m for m in range(6) if m not in (mX, mY)]
        assert np.isclose(XH[i], Xp[i, rest].mean()) and np.isclose(YH[i], Yp[i, rest].mean())
