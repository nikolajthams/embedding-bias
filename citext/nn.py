"""Small neural networks as the regressor of the residual regressions, for the experiments that are not linear by construction:
Experiment 3 (the local generator) and the application (author's request, 6 and 7 Oct 2026).

Regressor: sklearn MLPRegressor (rectified linear units, Adam, learning rate 1e-3, batches of 200, inputs standardized on the training
fold, early stopping on 15% of the training fold with a patience of 20 epochs, at most 500 epochs). Within each training fold the width,
depth and weight decay are chosen among CONFIGS by the validation score of the early-stopping split, so the test fold enters no choice,
and the prediction is the mean over SEEDS networks (different initializations and validation splits, each with its own choice). sklearn
divides the L2 term by the number of rows, so the penalties 10 and 100 are moderate: tuned on the Claude-generated data of the removed
experiment (6 Oct 2026), where the default penalty 1e-4 overfits the 384-dimensional inputs to a negative out-of-sample R^2 and these
settings match or slightly beat ridge regression. One network per target column and fold (no multi-task fitting), with seeds that differ
between the columns of one call, so that X and Y, fitted together as two columns, share no initialization; a shared seed correlates the
two prediction errors and inflates rho_hat.

Selection reuse: `select[j]` makes column j reuse the configurations chosen (per fold and seed) for column select[j] of the same call, and
`fixed[j]` passes configurations chosen in an earlier call, as info["chosen"][j] returns them. Experiment 3 chooses on the observed X and
Y and reuses the choice for the replicate draws and the null datasets (exp4_truth.py) and for the coupled alternatives (exp4_power.py).

Parallelism: one joblib task per (column, fold, seed, configuration), NN_JOBS workers (environment, default 8), each single-threaded.
Seeds and selection rule are those of experiments/nn_check.py (6 Oct 2026), whose single-call results this module reproduces exactly.
"""
from __future__ import annotations
import os, warnings
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONFIGS = {"full": [((32,), 10.0), ((128,), 10.0), ((64, 64), 10.0), ((32,), 100.0), ((128,), 100.0), ((64, 64), 100.0)],
           "small": [((64, 64), 10.0)]}
N_JOBS = int(os.environ.get("NN_JOBS", 8))
SEEDS = int(os.environ.get("NN_SEEDS", 3))


def mlp(hidden, alpha, seed):
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=1e-3, batch_size=200, early_stopping=True,
                                                        validation_fraction=0.15, n_iter_no_change=20, max_iter=500, random_state=seed))


def cfg_name(hidden, alpha): return f"{'x'.join(map(str, hidden))}/{alpha:g}"
def parse(name): h, a = name.split("/"); return tuple(int(v) for v in h.split("x")), float(a)
def column_seed(seed, j, k): return seed + 7919 * j + 101 * k


def _fit_one(Psi, y, tr, te, seed, hidden, alpha):
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = mlp(hidden, alpha, seed).fit(Psi[tr], y[tr])
        return m.predict(Psi[te]), float(m[-1].best_validation_score_), int(m[-1].n_iter_)


def crossfit_nn(Psi, Y, folds, configs="full", seed=0, select=None, fixed=None, n_jobs=None, seeds=None, verbose=0):
    """Cross-fitted network predictions of every column of Y ((n,) or (n, m)) on the given folds.
    configs: a key of CONFIGS or a list of (hidden_layer_sizes, alpha). select[j]: the column whose chosen configurations column j reuses
    (None: its own selection). fixed[j]: the configurations of column j from an earlier call (a list over folds of '+'-joined names, one
    per seed, as info['chosen'][j] gives them). Returns (predictions with the shape of Y, info) with info['chosen'] for every column,
    info['val_r2'] and info['iters'] (means over the networks used) and info['configs'] (the names used)."""
    Psi = np.ascontiguousarray(Psi, dtype=float); Y = np.asarray(Y, dtype=float); Y2 = np.ascontiguousarray(Y if Y.ndim == 2 else Y[:, None])
    cfgs = CONFIGS[configs] if isinstance(configs, str) else [(tuple(h), float(a)) for h, a in configs]
    n_jobs = N_JOBS if n_jobs is None else n_jobs; S = SEEDS if seeds is None else int(seeds); nc, nf = Y2.shape[1], len(folds)
    select = [None] * nc if select is None else list(select); fixed = {} if fixed is None else {int(j): list(v) for j, v in fixed.items()}
    free = [j for j in range(nc) if select[j] is None and j not in fixed]
    for j in range(nc):
        if j not in free and j not in fixed and select[j] not in free and select[j] not in fixed: raise ValueError(f"column {j} reuses column {select[j]}, which has no selection")
    chosen = {(j, k): fixed[j][k].split("+") for j in fixed for k in range(nf)}
    for j in fixed:
        if len(fixed[j]) != nf or any(len(chosen[(j, k)]) != S for k in range(nf)): raise ValueError(f"fixed configurations of column {j} do not match {nf} folds and {S} seeds")
    out = np.empty_like(Y2); val, iters = [], []

    def run(tasks):
        return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_fit_one)(Psi, Y2[:, j], folds[k][0], folds[k][1], column_seed(seed, j, k) + s, h, a) for j, k, s, h, a in tasks) if tasks else []
    # stage 0: the columns with a selection of their own; every configuration for every seed, the best validation score wins (the first of CONFIGS on ties)
    tasks = [(j, k, s, h, a) for j in free for k in range(nf) for s in range(S) for h, a in cfgs]; best = {}
    for (j, k, s, h, a), (pred, v, it) in zip(tasks, run(tasks)):
        if (j, k, s) not in best or v > best[(j, k, s)][0]: best[(j, k, s)] = (v, pred, cfg_name(h, a), it)
    for j in free:
        for k in range(nf):
            b = [best[(j, k, s)] for s in range(S)]; out[folds[k][1], j] = np.mean([x[1] for x in b], 0); chosen[(j, k)] = [x[2] for x in b]; val += [x[0] for x in b]; iters += [x[3] for x in b]
    # stage 1: the columns that reuse a selection or have fixed configurations
    rest = [j for j in range(nc) if j not in free]
    tasks = [(j, k, s, *parse((chosen[(j, k)] if j in fixed else chosen[(select[j], k)])[s])) for j in rest for k in range(nf) for s in range(S)]; acc = {}
    for (j, k, s, h, a), (pred, v, it) in zip(tasks, run(tasks)): acc.setdefault((j, k), []).append((s, pred, v, it, cfg_name(h, a)))
    for (j, k), L in acc.items():
        L.sort(key=lambda x: x[0]); out[folds[k][1], j] = np.mean([x[1] for x in L], 0); chosen[(j, k)] = [x[4] for x in L]; val += [x[2] for x in L]; iters += [x[3] for x in L]
    info = {"chosen": {j: ["+".join(chosen[(j, k)]) for k in range(nf)] for j in range(nc)}, "val_r2": float(np.mean(val)), "iters": float(np.mean(iters)),
            "configs": sorted({c for names in chosen.values() for c in names})}
    return out.reshape(Y.shape), info


def cfg_summary(info, cols=None):
    """'|'-joined names of the configurations used by the given columns (all columns by default)."""
    cols = range(len(info["chosen"])) if cols is None else cols
    return "|".join(sorted({c for j in cols for names in info["chosen"][j] for c in names.split("+")}))
