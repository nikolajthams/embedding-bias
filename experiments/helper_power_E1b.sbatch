#!/bin/bash
#SBATCH --job-name=power_E1b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/repo/logs/power_E1b_%j.out
# Helper job (7 Oct 2026): part of the power grid of E1 with the networks (targets 0.55 to 0.8), into its own results directory; the main job
# exp4_nn.sbatch covers the lower part. Merge the csv files into results/exp4_power_nn.csv afterwards.
R=/path/to/repo; PY=/path/to/venvs/ci_text/bin/python
export CI_TEXT_CODE_DIR=$R CI_TEXT_PAPER_DIR=$R/paper PYTHONPATH=$R HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CI_TEXT_RESULTS_DIR=$R/results_power_E1b
export NN_JOBS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
REGEN=$HOME/.cache/ci_text/llm/regen_qwen3-1.7b_n3000_M20.jsonl
cd $R; mkdir -p results_power_E1b
until [ -f $R/results/exp4_truth_nn.json ]; do sleep 60; done
echo "== power E1 0.55:0.8 $(date)"; $PY experiments/exp4_power.py $REGEN --observed hf --regressor mlp --views E1 --grid 0.55:0.8:0.025 --truth $R/results/exp4_truth_nn.csv --out $R/results_power_E1b
echo "== done $(date)"
