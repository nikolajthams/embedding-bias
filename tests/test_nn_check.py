"""Checks of the cross-fitted network regression of experiments/nn_check.py on synthetic data: the predictions are out of fold,
a nonlinear conditional mean is recovered better than by least squares, and a linear one about as well."""
import os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("CI_TEXT_CODE_DIR", HERE.parent)); sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "experiments")); sys.path.insert(0, str(HERE))
import nn_check as N
from citext.estimators import crossfit_folds, fit_predict_crossfit

def r2(y, m): return 1 - np.mean((y - m) ** 2) / np.var(y)

def make(n, d, seed, nonlinear):
    rng = np.random.default_rng(seed); Psi = rng.standard_normal((n, d)); u = Psi[:, :3].sum(1) / np.sqrt(3)
    f = np.sin(1.5 * u) + 0.5 * Psi[:, 3] * Psi[:, 4] if nonlinear else u
    return Psi, f + rng.standard_normal(n)

def test_nonlinear_mean_beats_least_squares():
    Psi, y = make(3000, 20, 0, True); folds = crossfit_folds(len(y), 5, 0)
    m, info = N.crossfit_nn(Psi, y, folds, N.CONFIGS["full"]); lin = fit_predict_crossfit(Psi, y, 5, 0, "ols", folds)
    assert r2(y, m) > r2(y, lin) + 0.05, (r2(y, m), r2(y, lin))
    assert info["val_r2"] > 0

def test_linear_mean_matches_least_squares():
    Psi, y = make(3000, 20, 1, False); folds = crossfit_folds(len(y), 5, 0)
    m, _ = N.crossfit_nn(Psi, y, folds, N.CONFIGS["full"]); lin = fit_predict_crossfit(Psi, y, 5, 0, "ols", folds)
    assert abs(r2(y, m) - r2(y, lin)) < 0.05, (r2(y, m), r2(y, lin))

def test_predictions_are_out_of_fold():
    """Perturbing the targets of one fold leaves the predictions of that fold unchanged."""
    Psi, y = make(1000, 10, 2, True); folds = crossfit_folds(len(y), 5, 0); te = folds[0][1]
    m1, _ = N.crossfit_nn(Psi, y, folds, N.CONFIGS["small"]); y2 = y.copy(); y2[te] += 10.0; m2, _ = N.crossfit_nn(Psi, y2, folds, N.CONFIGS["small"])
    assert np.allclose(m1[te], m2[te])
    assert not np.allclose(m1[folds[1][1]], m2[folds[1][1]])

def test_multi_target_shape_and_independence():
    Psi, y = make(600, 8, 3, True); Y = np.c_[y, -y]; folds = crossfit_folds(len(y), 5, 0)
    m, _ = N.crossfit_nn(Psi, Y, folds, N.CONFIGS["small"]); m0, _ = N.crossfit_nn(Psi, y, folds, N.CONFIGS["small"])
    assert m.shape == Y.shape and np.allclose(m[:, 0], m0)
