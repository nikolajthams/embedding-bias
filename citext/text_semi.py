"""Real text (20 Newsgroups) as Z with synthetic outcomes A, B (the paper's X, Y) whose dependence on Z
runs partly through information that the weak encoder E1 does not carry.

Design (all functions of Z, computed once on the full corpus and cached):
  E1 = all-MiniLM-L6-v2 (384), E2 = all-mpnet-base-v2 (768).
  Since 29 Sept 2026 (build_design(with_S=False), used by exp2_text.py): R = standardized E2,
  R_perp = residual of R after least squares on E1 over the corpus, g1 = standardized u'R_perp,
  g2 = standardized (v'R_perp orthogonalized against g1) with u, v random directions; so g1, g2
  are exactly linear in [E1, E2]. The original design (with_S=True) also had S = 100-dim TF-IDF
  SVD in R, a cross-fitted ridge residualization, and block-balanced directions.
  h_A, h_B = standardized first two principal components of E1 (the part that E1 does
  explain).
Outcomes for a subsample idx:
  A = h_A + c_A g1 + eps_A,  B = h_B + c_B (lam g1 + sqrt(1-lam^2) g2) + eps_B + beta A,
  c = sqrt(r2 / (1 - r2)), eps ~ N(0, 1).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .estimators import fit_predict_crossfit
from .paths import CACHE_DIR

MIN_WORDS = 30
TRUNC_WORDS = 300
E1_NAME = "all-MiniLM-L6-v2"
E2_NAME = "all-mpnet-base-v2"


def _safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name)


def load_corpus() -> dict:
    """Return {'texts': [...], 'target': [...], 'n_words': [...]} (cached)."""
    f = CACHE_DIR / "corpus_20ng.json"
    if f.exists():
        return json.loads(f.read_text())
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                              data_home=str(CACHE_DIR / "sklearn_data"))
    texts, target, n_words = [], [], []
    for t, y in zip(data.data, data.target):
        t = re.sub(r"\s+", " ", t).strip()
        w = t.split(" ")
        if len(w) >= MIN_WORDS:
            texts.append(" ".join(w[:TRUNC_WORDS]))
            target.append(int(y))
            n_words.append(min(len(w), TRUNC_WORDS))
    out = {"texts": texts, "target": target, "n_words": n_words}
    f.write_text(json.dumps(out))
    return out


def _device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def embed_texts(texts, model_name: str, cache_key: str | None = None, batch_size: int = 64):
    """Sentence-transformer embeddings, cached as .npy under ~/.cache/ci_text."""
    f = CACHE_DIR / f"emb_{cache_key or 'x'}_{_safe(model_name)}.npy" if cache_key else None
    if f is not None and f.exists():
        return np.load(f)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, cache_folder=str(CACHE_DIR / "st_models"), device=_device())
    model.max_seq_length = 256
    E = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=False)
    E = np.asarray(E, dtype=np.float64)
    if f is not None:
        np.save(f, E)
    return E


def tfidf_svd(texts, n_components: int = 100, cache_key: str = "corpus"):
    f = CACHE_DIR / f"svd{n_components}_{cache_key}.npy"
    if f.exists():
        return np.load(f)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(max_features=30000, sublinear_tf=True, min_df=3, stop_words="english").fit_transform(texts)
    S = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(X)
    np.save(f, S)
    return S


def standardize(M):
    M = np.asarray(M, dtype=float)
    return (M - M.mean(0)) / (M.std(0) + 1e-12)


def pca(M, k, seed=0):
    from sklearn.decomposition import PCA
    return PCA(n_components=k, random_state=seed).fit_transform(M)


def build_design(seed: int = 0, with_S: bool = True) -> dict:
    """Compute (and cache) everything that is a deterministic function of Z.

    with_S=True : the original design, R = standardized [E2, S] with S the 100 TF-IDF singular
                  directions, R_perp = R minus its cross-fitted ridge prediction from E1, and
                  block-balanced random directions (cache: design_seed{seed}.npz).
    with_S=False: the design of the paper since 29 Sept 2026, R = standardized E2 only and
                  R_perp = the residual of a least squares regression of R on E1 over the whole
                  corpus, so that g1, g2 are exactly linear in [E1, E2] and the 1152-dimensional
                  conditioning set [E1, E2] is sufficient (cache: design_seed{seed}_noS.npz)."""
    tag = "S" if with_S else "noS"
    f = CACHE_DIR / (f"design_seed{seed}.npz" if with_S else f"design_seed{seed}_noS.npz")
    corpus = load_corpus()
    texts = corpus["texts"]
    if f.exists():
        d = dict(np.load(f))
        d["texts"] = texts
        d["tag"] = tag
        return d
    E1 = embed_texts(texts, E1_NAME, cache_key="corpus")
    E2 = embed_texts(texts, E2_NAME, cache_key="corpus")
    rng = np.random.default_rng(seed + 1000)
    if with_S:
        S = tfidf_svd(texts, 100, "corpus")
        R = standardize(np.hstack([E2, S]))
        R_perp = R - fit_predict_crossfit(E1, R, K=5, seed=seed, regressor="ridge")
        dE2, dS = E2.shape[1], S.shape[1]

        def block_dir():
            a = rng.normal(size=dE2)
            b = rng.normal(size=dS)
            return np.concatenate([a / np.linalg.norm(a), b / np.linalg.norm(b)]) / np.sqrt(2)
    else:
        R = standardize(E2)
        X1 = np.column_stack([np.ones(len(E1)), E1])
        beta, *_ = np.linalg.lstsq(X1, R, rcond=None)
        R_perp = R - X1 @ beta

        def block_dir():
            a = rng.normal(size=E2.shape[1])
            return a / np.linalg.norm(a)

    u, v = block_dir(), block_dir()
    g1 = R_perp @ u
    g1 = (g1 - g1.mean()) / g1.std()
    g2 = R_perp @ v
    g2 = g2 - g1 * (g1 @ g2) / (g1 @ g1)
    g2 = (g2 - g2.mean()) / g2.std()
    PC = pca(E1, 2, seed)
    h_A = (PC[:, 0] - PC[:, 0].mean()) / PC[:, 0].std()
    h_B = (PC[:, 1] - PC[:, 1].mean()) / PC[:, 1].std()
    E1_32, E2_32 = pca(E1, 32, seed), pca(E2, 32, seed)
    d = {"E1": E1, "E2": E2, "E1_32": E1_32, "E2_32": E2_32, "g1": g1, "g2": g2,
         "h_A": h_A, "h_B": h_B, "target": np.asarray(corpus["target"])}
    if with_S:
        d["S"] = S
    np.savez(f, **d)
    d["texts"] = texts
    d["tag"] = tag
    return d


def scale(r2: float) -> float:
    return float(np.sqrt(r2 / (1 - r2))) if r2 > 0 else 0.0


def outcomes(design, idx, lam: float, r2_A: float, r2_B: float, beta: float, rng):
    n = len(idx)
    cA, cB = scale(r2_A), scale(r2_B)
    g1, g2 = design["g1"][idx], design["g2"][idx]
    A = design["h_A"][idx] + cA * g1 + rng.normal(size=n)
    B = design["h_B"][idx] + cB * (lam * g1 + np.sqrt(max(0.0, 1 - lam**2)) * g2) + rng.normal(size=n) + beta * A
    return A, B


def oracle_encoder(design, idx):
    """The 4-dim generating features: exactly sufficient for (A, B)."""
    return np.column_stack([design["h_A"][idx], design["h_B"][idx], design["g1"][idx], design["g2"][idx]])


def population_quantities(design, lam: float, r2_A: float, r2_B: float, regressor: str = "krr",
                          K: int = 5, seed: int = 0, n_max: int | None = None) -> dict:
    """Corpus-level ('population') quantities for psi = E1, with the conditional
    expectations E[. | E1] estimated by cross-fitting on the whole corpus:
    R2_A, R2_B, rho, bound = sqrt(R2_A R2_B) and the population benchmark for E2."""
    name = f"pop_{regressor}_lam{lam}_r2{r2_A}_{r2_B}_seed{seed}_nmax{n_max}.json"
    if design.get("tag", "S") != "S":            # the cache is per design variant
        name = f"pop_{design['tag']}_" + name[4:]
    f = CACHE_DIR / name
    if f.exists():
        return json.loads(f.read_text())
    rng = np.random.default_rng(seed)
    n_all = len(design["g1"])
    idx = np.arange(n_all) if n_max is None or n_max >= n_all else rng.choice(n_all, n_max, replace=False)
    cA, cB = scale(r2_A), scale(r2_B)
    g1, g2 = design["g1"][idx], design["g2"][idx]
    gB = lam * g1 + np.sqrt(max(0.0, 1 - lam**2)) * g2
    E1, E2 = design["E1"][idx], design["E2"][idx]
    # the h parts are exact functions of E1 and drop out of the residuals of a
    # sufficiently flexible regression; include them for realism
    fA = design["h_A"][idx] + cA * g1
    fB = design["h_B"][idx] + cB * gB
    F = np.column_stack([fA, fB])
    res1 = F - fit_predict_crossfit(E1, F, K, seed, regressor)          # Delta given E1
    res12 = F - fit_predict_crossfit(np.hstack([E1, E2]), F, K, seed, regressor)
    var_dA, var_dB = res1[:, 0].var(), res1[:, 1].var()
    tau = np.cov(res1[:, 0], res1[:, 1])[0, 1]
    sA2, sB2 = var_dA + 1.0, var_dB + 1.0
    r2A, r2B = var_dA / sA2, var_dB / sB2
    bench_A = (sA2 - (res12[:, 0].var() + 1.0)) / sA2
    bench_B = (sB2 - (res12[:, 1].var() + 1.0)) / sB2
    out = {"r2_A": float(r2A), "r2_B": float(r2B), "rho": float(tau / np.sqrt(sA2 * sB2)),
           "bound": float(np.sqrt(r2A * r2B)), "bench_r2_A": float(max(0, bench_A)),
           "bench_r2_B": float(max(0, bench_B)),
           "bench": float(np.sqrt(max(0, bench_A) * max(0, bench_B))), "regressor": regressor,
           "n_used": int(len(idx))}
    f.write_text(json.dumps(out))
    return out
