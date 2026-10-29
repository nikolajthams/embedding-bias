"""Linear-Gaussian (and tanh) latent model with a controllable representation gap.

Z ~ N(0, I_p); psi_q(Z) = Z[:, :q] keeps the first q coordinates. Outcomes (A, B are the
paper's tested variables X, Y)
A = f_A(Z) + eps_A, B = f_B(Z) + eps_B (+ beta * A under H1), unit Gaussian noise.
The coefficients on the dropped coordinates are c_A u for A and
c_B (lam u + sqrt(1 - lam^2) v) for B with u, v orthonormal, so that under H0
    rho_psi = lam * sqrt(R2_A R2_B)   (R2_A, R2_B: the paper's R2_psi(X), R2_psi(Y)),
i.e. the bound of the paper is attained at lam = 1 and the bias vanishes at
lam = 0 although psi_q is insufficient.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import integrate

# E[tanh(Z)^2] for Z ~ N(0, 1)
KAPPA_TANH = integrate.quad(lambda z: np.tanh(z) ** 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
                            -12, 12)[0]


@dataclass(frozen=True)
class LinearConfig:
    p: int = 20
    q: int = 5           # kept coordinates; q = p is the sufficient representation
    r2_A: float = 0.1
    r2_B: float = 0.1
    lam: float = 1.0
    beta: float = 0.0    # H1 effect: B += beta * A
    nonlinear: bool = False
    seed_dirs: int = 12345

    def with_(self, **kw):
        return replace(self, **kw)


def _dropped_directions(cfg: LinearConfig):
    d = cfg.p - cfg.q
    if d == 0:
        return None, None
    rng = np.random.default_rng(cfg.seed_dirs)
    if d == 1:
        return np.ones(1), np.zeros(1)
    M = rng.normal(size=(d, 2))
    Q, _ = np.linalg.qr(M)
    return Q[:, 0], Q[:, 1]


def _scale(r2: float, nonlinear: bool) -> float:
    """c such that Var(c * g) / (Var(c * g) + 1) = r2 with Var(g) = 1 (linear) or
    KAPPA_TANH (tanh)."""
    if r2 <= 0:
        return 0.0
    var_g = KAPPA_TANH if nonlinear else 1.0
    return float(np.sqrt(r2 / ((1 - r2) * var_g)))


def coefficients(cfg: LinearConfig):
    """Return (beta_A, beta_B), each of length p."""
    p, q = cfg.p, cfg.q
    beta_A = np.zeros(p)
    beta_B = np.zeros(p)
    # kept part: two different fixed patterns of norm 1 (explained by psi_q)
    beta_A[:q] = 1.0 / np.sqrt(q)
    signs = np.where(np.arange(q) % 2 == 0, 1.0, -1.0)
    beta_B[:q] = signs / np.sqrt(q)
    u, v = _dropped_directions(cfg)
    if u is not None:
        c_A, c_B = _scale(cfg.r2_A, cfg.nonlinear), _scale(cfg.r2_B, cfg.nonlinear)
        beta_A[q:] = c_A * u
        beta_B[q:] = c_B * (cfg.lam * u + np.sqrt(max(0.0, 1 - cfg.lam**2)) * v)
    return beta_A, beta_B


def _features(Z, nonlinear):
    return np.tanh(Z) if nonlinear else Z


def sample(cfg: LinearConfig, n: int, rng: np.random.Generator):
    """Draw n observations. Returns dict with Z, Psi, A, B and the noiseless parts."""
    beta_A, beta_B = coefficients(cfg)
    Z = rng.normal(size=(n, cfg.p))
    F = _features(Z, cfg.nonlinear)
    fA, fB = F @ beta_A, F @ beta_B
    A = fA + rng.normal(size=n)
    B = fB + rng.normal(size=n) + cfg.beta * A
    return {"Z": Z, "Psi": Z[:, : cfg.q], "A": A, "B": B, "fA": fA, "fB": fB}


def conditional_means_given_psi(cfg: LinearConfig, Z):
    """Population E[A | psi_q(Z)] and E[B | psi_q(Z)] (used only for checks).
    Coordinates are independent with E[Z_j] = E[tanh(Z_j)] = 0, so the dropped
    part integrates to zero."""
    beta_A, beta_B = coefficients(cfg)
    F = _features(Z, cfg.nonlinear)
    mA = F[:, : cfg.q] @ beta_A[: cfg.q]
    mB = F[:, : cfg.q] @ beta_B[: cfg.q]
    # under H1, E[B | psi] = E[fB | psi] + beta E[A | psi]
    return mA, mB + cfg.beta * mA


def population(cfg: LinearConfig) -> dict:
    """Analytic population quantities: R2_A, R2_B, tau, rho, bound = sqrt(R2_A R2_B)."""
    beta_A, beta_B = coefficients(cfg)
    var_g = KAPPA_TANH if cfg.nonlinear else 1.0
    dA, dB = beta_A[cfg.q:], beta_B[cfg.q:]
    var_dA = var_g * float(dA @ dA)        # Var(Delta_A)
    var_dB = var_g * float(dB @ dB)
    tau0 = var_g * float(dA @ dB)          # E[Delta_A Delta_B] under H0
    sA2 = var_dA + 1.0                     # sigma_A^2
    # H1: r_B = Delta_B + eps_B + beta r_A
    tau = tau0 + cfg.beta * sA2
    sB2 = var_dB + 1.0 + cfg.beta**2 * sA2 + 2 * cfg.beta * tau0
    r2_A = var_dA / sA2
    r2_B_H0 = var_dB / (var_dB + 1.0)
    rho = tau / np.sqrt(sA2 * sB2)
    return {"r2_A": r2_A, "r2_B": r2_B_H0, "tau": tau, "rho": float(rho),
            "bound": float(np.sqrt(r2_A * r2_B_H0)), "sigma_A": float(np.sqrt(sA2)),
            "sigma_B": float(np.sqrt(sB2))}


def partial_r2_population(cfg: LinearConfig, q_prime: int) -> dict:
    """Population partial R^2 of psi_{q'} beyond psi_q (q' > q) for A and B in the
    linear model: the share of Var(Delta) carried by coordinates q..q'-1."""
    beta_A, beta_B = coefficients(cfg)
    var_g = KAPPA_TANH if cfg.nonlinear else 1.0
    out = {}
    for name, b in (("A", beta_A), ("B", beta_B)):
        d = b[cfg.q:]
        gain = var_g * float(d[: q_prime - cfg.q] @ d[: q_prime - cfg.q])
        out[f"r2_{name}"] = gain / (var_g * float(d @ d) + 1.0)
    out["bench"] = float(np.sqrt(out["r2_A"] * out["r2_B"]))
    return out
