#!/bin/bash
#SBATCH --job-name=nn_check
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/repo/logs/nn_check_%j.out
# Neural network check (nn_check.py) on CPU: Experiments 3 and 4 and the application, then the macro block (printed).
R=/path/to/repo
mkdir -p $HOME/.cache; [ -e $HOME/.cache/ci_text ] || ln -s $R/cache $HOME/.cache/ci_text
cd $R || exit 1
export CI_TEXT_CODE_DIR=$R CI_TEXT_PAPER_DIR=$R/paper PYTHONPATH=$R HF_HUB_OFFLINE=1 NN_JOBS=46 OMP_NUM_THREADS=1
PY=/path/to/venvs/ci_text/bin/python
for ph in exp3 exp4 app macros; do echo "== $ph $(date)"; $PY experiments/nn_check.py --only $ph 2>&1 | grep -v Warning; done
echo "== done $(date)"
