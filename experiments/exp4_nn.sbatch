#!/bin/bash
#SBATCH --job-name=exp4_nn
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/path/to/repo/logs/exp4_nn_%j.out
# Experiment 3 (local generator) with the small networks of citext/nn.py as the regressor (7 Oct 2026): identification and level on the
# resampled null datasets (exp4_truth.py --regressor mlp, embeddings of 64 to 768 coordinates) and power against the coupled alternatives
# (exp4_power.py --regressor mlp, resumable). CPU only, about 5 h on 128 cores. Staged layout: /path/to/ci_text/{code,paper},
# the cache in ~/.cache/ci_text (embeddings, hidden states, the regenerated draws), the venv /path/to/venvs/ci_text.
# Afterwards copy results/exp4_truth_nn* and exp4_power_nn* back and run fig_exp4.py and the two scripts' --write locally.
R=/path/to/repo; PY=/path/to/venvs/ci_text/bin/python
export CI_TEXT_CODE_DIR=$R CI_TEXT_PAPER_DIR=$R/paper PYTHONPATH=$R HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NN_JOBS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
REGEN=$HOME/.cache/ci_text/llm/regen_qwen3-1.7b_n3000_M20.jsonl
cd $R
echo "== truth $(date)"; $PY experiments/exp4_truth.py $REGEN --observed hf --regressor mlp --dims 64,128,256,384,768 --no-plot
echo "== power $(date)"; $PY experiments/exp4_power.py $REGEN --observed hf --regressor mlp
echo "== done $(date)"
