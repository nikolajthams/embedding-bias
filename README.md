# Embedding-Bias in Conditional Independence Testing

Code and data for the paper [*Embedding-Bias in Conditional Independence Testing*](https://arxiv.org/abs/2610.11584) (2026).

To test whether $X$ and $Y$ are conditionally independent given a text $Z$, one conditions on an embedding $\psi(Z)$ in place of $Z$.
The embedded test is valid if $Z$ is independent of $X$ or of $Y$ given $\psi(Z)$, which cannot be confirmed from data, and when this
fails the rejection probability under the null hypothesis can tend to one. The paper treats the information that the embedding
discards as an omitted variable. For a residual correlation test (in the spirit of the Generalised Covariance Measure), the bias
under the null hypothesis equals the correlation of the parts of $\mathbb{E}[X \mid Z]$ and $\mathbb{E}[Y \mid Z]$ that the
embedding misses, times the geometric mean of two partial $R^2$ values. This identity gives

- a **robust test** that is valid under a declared tolerance $B$ for that geometric mean (`robust_test`, `critical_value`),
- a **robustness value** $\mathrm{RV}_\alpha$, the smallest tolerance at which a rejection goes away (`rv_alpha`), and
- a **demonstrated insufficiency**: a lower bound on the bias from a richer embedding $\psi'$ (`benchmark_pair`).

The repository contains the package `citext` that implements these, the scripts of the four experiments of the paper, their
result tables, the generated and rated data they need, and the figures and numbers that the paper uses.

## Example

```python
import numpy as np
from citext.synthetic import LinearConfig, sample, population
from citext.estimators import gcm_test, robust_test, rv_alpha, benchmark_pair

# Z in R^20; the embedding psi(Z) keeps 5 coordinates; X and Y depend on the dropped coordinates in an aligned way
cfg = LinearConfig(p=20, q=5, r2_A=0.1, r2_B=0.1, lam=1.0)
d = sample(cfg, n=4000, rng=np.random.default_rng(0))

res = gcm_test(d["A"], d["B"], d["Psi"], regressor="ols")   # residual-correlation test on psi(Z); X, Y are A, B in the code
print(f"rho_hat = {res.rho_hat:.3f} (se {res.se:.3f}), p = {res.pvalue:.2g}")   # rejects although X and Y are independent given Z
print(f"bias bound sqrt(R2_A R2_B) = {population(cfg)['bound']:.3f}")

print(robust_test(res.rho_hat, res.se, tol=0.1))       # robust test with insufficiency tolerance B = 0.1
print(f"RV_alpha = {rv_alpha(res.rho_hat, res.se):.3f}")   # robustness value: smallest tolerance at which the test no longer rejects

b = benchmark_pair(d["A"], d["B"], d["Psi"], d["Z"][:, 5:10], regressor="ols")   # demonstrated insufficiency of psi against a richer psi'
print(f"demonstrated insufficiency rho_bench = {b['bench']:.3f}")
```

```
rho_hat = 0.086 (se 0.016), p = 4.5e-08
bias bound sqrt(R2_A R2_B) = 0.100
{'reject': False, 'c': 0.0260, 'tol': 0.1, 'alpha': 0.05, 'stat': -0.869}
RV_alpha = 0.060
demonstrated insufficiency rho_bench = 0.027
```

## Installation

Python 3.12 with the pinned versions of `requirements.txt` (frozen on macOS arm64; a CUDA build of PyTorch is needed for the
generation step of Experiment 3):

```bash
uv venv --python 3.12 ~/.venvs/ci_text
uv pip install --python ~/.venvs/ci_text/bin/python -r requirements.txt
export PYTHONPATH=.            # run everything from the repository root
~/.venvs/ci_text/bin/python -m pytest -q tests
```

Large artifacts (corpora, encoder weights, embeddings, generated texts, ratings) are cached in `~/.cache/ci_text/`.
Steps that call a commercial API read the keys from a `.env` file at the repository root (`ANTHROPIC_API_KEY`,
`TYPESAFE_API_KEY`, `OPENAI_API_KEY`); it is ignored by git and never printed.

## Layout

```
citext/            the package
  estimators.py    cross-fitted residuals; the standard test (gcm_test); the robust test (robust_test, critical_value, rv_alpha);
                   the demonstrated insufficiency (partial_r2_benchmark, benchmark_pair); regressors (make_regressor, KRRMedian)
  synthetic.py     linear-Gaussian latent model of Experiment 1 with an alignment parameter lam; analytic population quantities
  text_semi.py     20 Newsgroups as Z; the encoders E1 = all-MiniLM-L6-v2 and E2 = all-mpnet-base-v2; synthetic outcomes
  llm_gen.py       generation of (Z, X, Y) with the Anthropic API (the source of the 3000 posts of Experiment 3)
  nn.py            small neural networks as an optional regressor (not used in the paper)
  plotting.py      figure style;  paths.py: result, figure and cache directories
experiments/       one script per experiment phase (see the table below) and Slurm templates (*.sbatch) for the GPU and
                   multi-core steps; set the account and paths before use
results/           the result tables (csv/json) that the figures and the numbers of the paper are made from
data/              the generated texts and the politeness ratings (see Data)
paper/figures/     the figures of the paper, as written by the scripts;  paper/results_macros.tex: every number of the paper
                   as a LaTeX macro, written by the *_macros.py scripts and by exp4_*.py and app_cmv_deep.py with --write
tests/             pytest checks of the analytic formulas, the calibration of the robust test, the level of the standard test,
                   and the replicate-based identification and the coupled alternatives of Experiment 3
run_all.sh         the whole pipeline in order
```

## Reproducing the paper

The scripts are numbered in the order in which the experiments were developed, which differs from the paper: the code's
*Experiment 4* is **Experiment 3 of the paper**, and the code's *Experiment 3* is not in the paper (see below).

| Paper | Scripts | Output |
|---|---|---|
| Section 4.1, Experiment 1: Gaussian data | `exp1_synthetic.py 500`, `exp1_macros.py --write` | `results/exp1_*`, `summary_exp1.json`; `exp1_power.pdf`, `exp1_bound_tightness.pdf`, `exp1_rv_estimation.pdf`; `\resOne*` |
| Section 4.2, Experiment 2: real text with synthetic outcomes | `exp2_text.py 100`, `exp2_macros.py --write` | `results/exp2_*`, `summary_exp2.json`; `exp2_text_power.pdf`, `exp2_text_rv_benchmark.pdf`; `\resTwo*` |
| Type I error panel of Experiments 1 and 2 | `fig_exp123.py` | `figures/exp123_main.tex` (TikZ, input by the paper) |
| Section 4.3, Experiment 3: text generated by a local model | `exp4_local.py`, `exp4_regen.py`, `exp4_truth.py`, `exp4_power.py`, `fig_exp4.py` | `results/exp4_*`; `exp4_rejection.pdf`, `exp4_main.pdf`, `exp4_power_views.pdf`, `exp4_identity.pdf`; `\resFour*` |
| Section 4.4, Application: replies in ChangeMyView | `app_cmv_deep.py --rater jev --write` (helpers in `app_cmv.py`) | `results/app_cmv_deep_*`; `app_cmv_deep_main_jev.pdf`, `app_cmv_deep_views_jev.pdf`; `\resDeep*` |

Every number in the paper is a macro in `paper/results_macros.tex`, written by the scripts from the files in `results/`, and
every figure is written by the scripts to `paper/figures/` (set `CI_TEXT_PAPER_DIR` to write them elsewhere). Seeds are fixed
in the scripts, folds use `KFold`/`GroupKFold` with fixed seeds, and the generated texts and ratings are cached, so the analyses
are deterministic given the data below.

```bash
PY=~/.venvs/ci_text/bin/python; export PYTHONPATH=.
mkdir -p ~/.cache/ci_text && cp -R data/llm data/cmv ~/.cache/ci_text/        # the shipped data (see Data)

$PY experiments/exp1_synthetic.py 500 && $PY experiments/exp1_macros.py --write     # minutes
$PY experiments/exp2_text.py 100 && $PY experiments/exp2_macros.py --write          # about 1 h; downloads 20 Newsgroups and the encoders
$PY experiments/fig_exp123.py

# Experiment 3 of the paper (code: exp4). With the shipped generations, only the hidden states need a GPU (minutes); the
# generation itself (`--only gen`, about 3.5 h on a laptop GPU) and the replicates (exp4_regen.sbatch, minutes on one H100) are optional.
$PY experiments/exp4_local.py --only hidden,test,plots --write
$PY experiments/exp4_truth.py ~/.cache/ci_text/llm/regen_qwen3-1.7b_n3000_M20.jsonl --observed hf --write   # identification, robust test at the true tolerance
$PY experiments/exp4_power.py ~/.cache/ci_text/llm/regen_qwen3-1.7b_n3000_M20.jsonl --observed hf --write   # power against coupled alternatives
$PY experiments/fig_exp4.py

# Application. `build` downloads the ConvoKit corpus, `embed` runs the encoders (CPU is fine); the ratings are shipped,
# so `rate` makes no API calls (and is skipped without a key) and `agree` uses the shipped ratings.
$PY experiments/app_cmv_deep.py --rater jev --write
```

`run_all.sh` runs the same steps in order (and the API steps when the keys are set).

## Data

The folder `data/` holds the generated and rated data, so that the experiments can be reproduced without API access or
hours of GPU time. Copy it into the cache before running (`cp -R data/llm data/cmv ~/.cache/ci_text/`).

- `data/llm/gen_claude-haiku-4-5_n3000.jsonl`: the 3000 posts from 20 Newsgroups that are $Z$ in Experiment 3, with the texts
  that Claude Haiku 4.5 generated from them (`citext.llm_gen`; only the posts are used in the paper).
- `data/llm/gen_qwen3-1.7b_n3000.jsonl`: the observed comment and summary per post from Qwen3-1.7B (`exp4_local.py --only gen`);
  the hidden states are recomputed from it with `exp4_local.py --only hidden` (needs the Qwen3-1.7B weights).
- `data/llm/regen_qwen3-1.7b_n3000_M20.jsonl`: the $M = 20$ regenerated comments and summaries per post (`exp4_regen.py`), from
  which the partial $R^2$ values, $\lambda$ and the true insufficiency are identified.
- `data/cmv/ratings_{jev,haiku,openai}_deep.jsonl`: politeness ratings of the replies (one line per reply id) for the
  application. The paper uses the Jev ratings; the Claude Haiku 4.5 ratings are the agreement check in the appendix on
  implementation details; the GPT-5 nano ratings are an alternative rater that is not used in the paper.
- 20 Newsgroups is downloaded by scikit-learn on first use. The ChangeMyView corpus (the ConvoKit `winning-args-corpus` of
  Tan et al., 2016) is downloaded by `app_cmv_deep.py --only build` from
  https://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/.

## Notation: code versus paper

- The paper's tested variables $X$ and $Y$ are `A` and `B` in the code and in the result files (`rA`, `rB` residuals;
  `r2_A`, `r2_B`, `R2_A`, `R2_B` for the partial $R^2$ parameters $R^2_\psi(X)$, $R^2_\psi(Y)$). In Experiment 3,
  `X`, `Y0`, `Y1` are the generated texts (comment, summary under $H_0$, summary under $H_1$) and `A`, `B0`, `B1` their sentiment scores.
- The insufficiency tolerance $B$ of the paper is the argument `tol` of `critical_value` and `robust_test`.
- The number $M$ of regenerations per post is `--M` in `exp4_regen.py` and `--B` in `citext.llm_gen`.
- `rho`, `rho_hat` estimate $\rho_\psi$; `bench`, `rho_bench` is the demonstrated insufficiency $\rho_{\mathrm{bench}}(\psi \to \psi')$.
  Figure labels use the paper's notation.

## Not in the paper

Two things were tried during the project and are kept for the record only.

- The code's Experiment 3 (`experiments/exp3_*.py`, `results/exp3_*`, `summary_exp3.json`): the same design with text
  generated by Claude Haiku 4.5 and an exact model-X reference. Experiment 3 of the paper, with a local generator whose states
  are accessible, covers the same point. `exp3_macros.py` still supplies three constants of the paper (the number of posts, the
  generating model and the truncation share). See `results/OBSOLETE_exp3.md`.
- Small neural networks as the regressor (`citext/nn.py`, `--regressor mlp` of `exp4_truth.py`, `exp4_power.py` and
  `app_cmv_deep.py`, files `*_nn*`, `nn_check.py`, the Slurm jobs `exp4_nn.sbatch`, `app_nn.sbatch` and `helper_*.sbatch`).
  The paper uses least squares throughout Experiment 3 and the application.

The earlier application design in `app_cmv.py` (replies to the post rather than to a comment) is kept because
`app_cmv_deep.py` builds on its helpers; `app_cmv_macros.py` supplies the number of posts in the corpus.

## Citation

```bibtex
@article{thams2026embedding,
  title         = {Embedding-Bias in Conditional Independence Testing},
  author        = {Thams, Nikolaj and Lundborg, Anton Rask},
  journal       = {arXiv preprint arXiv:2610.11584},
  year          = {2026},
  eprint        = {2610.11584},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}
```

## License

The code is released under the MIT License (see `LICENSE`). Qwen3-1.7B and the two sentence encoders are released under
Apache 2.0; the 20 Newsgroups and ChangeMyView corpora are distributed by their creators for research use; the texts and
ratings in `data/` were obtained through the providers' APIs under their terms of use.
