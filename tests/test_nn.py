"""Checks of the cross-fitted network regression of citext.nn on synthetic data: the predictions are out of fold, a nonlinear conditional
mean is recovered better than by least squares and a linear one about as well, the columns of one call are independent, and a selection
passed back through `fixed` reproduces the predictions of `select`."""
import os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("CI_TEXT_CODE_DIR", HERE.parent)); sys.path.insert(0, str(CODE))
from citext.nn import crossfit_nn, CONFIGS  # noqa: E402
from citext.estimators import crossfit_folds, fit_predict_crossfit  # noqa: E402

def r2(y, m): return 1 - np.mean((y - m) ** 2) / np.var(y)

def make(n, d, seed, nonlinear):
    rng = np.random.default_rng(seed); Psi = rng.standard_normal((n, d)); u = Psi[:, :3].sum(1) / np.sqrt(3)
    f = np.sin(1.5 * u) + 0.5 * Psi[:, 3] * Psi[:, 4] if nonlinear else u
    return Psi, f + rng.standard_normal(n)

def test_nonlinear_mean_beats_least_squares():
    Psi, y = make(3000, 20, 0, True); folds = crossfit_folds(len(y), 5, 0)
    m, info = crossfit_nn(Psi, y, folds, "full"); lin = fit_predict_crossfit(Psi, y, 5, 0, "ols", folds)
    assert r2(y, m) > r2(y, lin) + 0.05, (r2(y, m), r2(y, lin))
    assert info["val_r2"] > 0 and len(info["chosen"][0]) == 5

def test_linear_mean_matches_least_squares():
    Psi, y = make(3000, 20, 1, False); folds = crossfit_folds(len(y), 5, 0)
    m, _ = crossfit_nn(Psi, y, folds, "full"); lin = fit_predict_crossfit(Psi, y, 5, 0, "ols", folds)
    assert abs(r2(y, m) - r2(y, lin)) < 0.05, (r2(y, m), r2(y, lin))

def test_predictions_are_out_of_fold():
    """Perturbing the targets of one fold leaves the predictions of that fold unchanged."""
    Psi, y = make(1000, 10, 2, True); folds = crossfit_folds(len(y), 5, 0); te = folds[0][1]
    m1, _ = crossfit_nn(Psi, y, folds, "small"); y2 = y.copy(); y2[te] += 10.0; m2, _ = crossfit_nn(Psi, y2, folds, "small")
    assert np.allclose(m1[te], m2[te])
    assert not np.allclose(m1[folds[1][1]], m2[folds[1][1]])

def test_columns_are_independent_and_seeded_apart():
    Psi, y = make(600, 8, 3, True); Y = np.c_[y, -y]; folds = crossfit_folds(len(y), 5, 0)
    m, _ = crossfit_nn(Psi, Y, folds, "small"); m0, _ = crossfit_nn(Psi, y, folds, "small")
    assert m.shape == Y.shape and np.allclose(m[:, 0], m0)
    assert not np.allclose(m[:, 1], -m[:, 0])          # the second column has its own initialization and validation split

def test_fixed_reproduces_select():
    cf = [((16,), 10.0), ((32,), 10.0)]; Psi, y = make(600, 8, 4, True); Y = np.c_[y, y + 0.1]; folds = crossfit_folds(len(y), 5, 0)
    m_sel, info = crossfit_nn(Psi, Y, folds, cf, select=[None, 0])
    assert info["chosen"][1] == info["chosen"][0]
    m_fix, info2 = crossfit_nn(Psi, Y, folds, cf, fixed={1: info["chosen"][0]})
    assert np.allclose(m_sel[:, 1], m_fix[:, 1]) and info2["chosen"][1] == info["chosen"][0]
