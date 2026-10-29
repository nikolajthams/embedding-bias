#!/bin/bash
#SBATCH --job-name=app_main
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/repo/logs/app_main_%j.out
# Helper job (7 Oct 2026): the application with the networks for E1 only, the six main pairs, no benchmarks (what the main-text figure needs),
# into its own results directory; the rows equal those of the full run of app_nn.sbatch (same data, folds and seeds).
R=/path/to/repo; PY=/path/to/venvs/ci_text/bin/python
export CI_TEXT_CODE_DIR=$R CI_TEXT_PAPER_DIR=$R/paper PYTHONPATH=$R HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CI_TEXT_RESULTS_DIR=$R/results_main
export NN_JOBS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $R; mkdir -p results_main
echo "== app E1 $(date)"; $PY experiments/app_cmv_deep.py --rater jev --regressor mlp --only test --views E1 --no-bench
echo "== done $(date)"
