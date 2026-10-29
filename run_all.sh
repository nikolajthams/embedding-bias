#!/usr/bin/env bash
# Reproduce all experiments, figures and result files. Environment: ~/.venvs/ci_text (see README.md, requirements.txt).
# Steps that call the Anthropic API (Experiment 3 generation, model-X reference, politeness rating) need ANTHROPIC_API_KEY
# in .env at the repository root; they are resumable and cached under ~/.cache/ci_text/. Steps that only re-plot saved results are cheap.
set -euo pipefail
cd "$(dirname "$0")"
PY=~/.venvs/ci_text/bin/python
export PYTHONPATH=.
$PY -m pytest -q tests
$PY experiments/exp1_synthetic.py 500                       # Experiment 1 (minutes)
$PY experiments/exp2_text.py 100                            # Experiment 2 (about 1 h; downloads 20 Newsgroups and the encoders)
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  $PY -m citext.llm_gen --n 3000 --model claude-haiku-4-5    # Experiment 3 data (about 6 USD, 25 min)
  # Experiment 3 (exp3_llm.py, exp3_crt.py, exp3_diagnostics.py) is OBSOLETE since 7 Oct 2026, see results/OBSOLETE_exp3.md
  # $PY experiments/exp3_llm.py ~/.cache/ci_text/llm/gen_claude-haiku-4-5_n3000.jsonl 100
  $PY -m citext.llm_gen --crt ~/.cache/ci_text/llm/gen_claude-haiku-4-5_n3000.jsonl --n-rows 500 --B 19 --model claude-haiku-4-5   # model-X reference draws (about 6 USD)
  # $PY experiments/exp3_crt.py ~/.cache/ci_text/llm/gen_claude-haiku-4-5_n3000.jsonl ~/.cache/ci_text/llm/crt_claude-haiku-4-5_n500_B19.jsonl
  # $PY experiments/exp3_diagnostics.py                       # learning curve, CRT figure, truncation, views/benchmarks
  $PY experiments/app_cmv.py                                # application: build, embed, rate (about 7.5 USD), test, plots
  $PY experiments/app_cmv_mainfig.py
  $PY experiments/app_cmv_components.py
  $PY experiments/app_cmv_deep.py --rater jev --write    # application, parent-comment design (Section 4.4): build, embed, rate (Jev), test, components, plots, macros
else
  echo "ANTHROPIC_API_KEY not set: skipping Experiment 3 and the application"
fi
$PY experiments/exp4_local.py                               # Experiment 4: local generator (laptop GPU, about 3.5 h), hidden states, tests, figure, macros
# Experiment 4 replicates need a GPU: on a Slurm cluster `sbatch experiments/exp4_regen.sbatch` (one H100, 7 min), or on this machine
# $PY experiments/exp4_regen.py --engine vllm --model <Qwen3-1.7B dir> --M 20 --source ~/.cache/ci_text/llm/gen_claude-haiku-4-5_n3000.jsonl --out-dir ~/.cache/ci_text/llm
REGEN=~/.cache/ci_text/llm/regen_qwen3-1.7b_n3000_M20.jsonl
$PY experiments/exp4_truth.py $REGEN --observed hf --write   # identification, robust test at the true tolerance (least squares; minutes)
$PY experiments/exp4_power.py $REGEN --observed hf --write   # power against coupled alternatives (30 s)
$PY experiments/fig_exp4.py                                 # Experiment 3 main-text and appendix figures
# Optional comparison, not in the paper: the same with the small networks of citext/nn.py (`--regressor mlp`, files *_nn, macros \resFourTruthNn*/\resFourPowerNn*;
# hours on many cores, Slurm jobs exp4_nn.sbatch and app_nn.sbatch): tried and dropped on 7 Oct 2026.
$PY experiments/regen_figs.py                               # appendix figures from saved results
$PY experiments/collect_summary.py
