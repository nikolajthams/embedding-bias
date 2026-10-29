#!/bin/bash
#SBATCH --job-name=app_nn
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/repo/logs/app_nn_%j.out
# Memory: 128 workers each copy the 49,050 x 1,152 matrix of [E1,E2] (a job with 128G was killed there), hence 400G.
# Application (parent-comment design) with the small networks of citext/nn.py as the regressor (7 Oct 2026): the test phase of
# app_cmv_deep.py --regressor mlp (six main pairs, resumable per pair), about 1.5 h on 128 cores, CPU only. Needs the staged cache
# ~/.cache/ci_text/cmv (rows_deep.parquet, parents_deep.parquet, views_deep.npz, ratings_jev_deep.jsonl). Afterwards copy
# results/app_cmv_deep_results_jev_nn.csv and app_cmv_deep_bench_jev_nn.csv back and run --only components,plots,macros --write locally.
R=/path/to/repo; PY=/path/to/venvs/ci_text/bin/python
export CI_TEXT_CODE_DIR=$R CI_TEXT_PAPER_DIR=$R/paper PYTHONPATH=$R HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NN_JOBS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $R
echo "== app $(date)"; $PY experiments/app_cmv_deep.py --rater jev --regressor mlp --only test
echo "== done $(date)"
