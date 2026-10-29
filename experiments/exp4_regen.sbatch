#!/bin/bash
# Experiment 4, ground-truth tolerance: regenerate the comment and the summary M times per post with Qwen3-1.7B (bf16) under
# vLLM on one H100 (about 15 to 30 minutes for M = 20). Resumable: resubmitting continues from the draws already written.
# Submit:  sbatch exp4_regen.sbatch            (M defaults to 20; override with  sbatch --export=ALL,M=40 exp4_regen.sbatch)
#SBATCH --job-name=exp4-regen
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/repo/logs/%x-%j.out
set -euo pipefail
WORK=/path/to/repo
MODEL=/path/to/models/Qwen3-1.7B
IMAGE=/path/to/containers/vllm-openai-v0.29.0.sqsh
M=${M:-20}
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
echo "job $SLURM_JOB_ID on $(hostname), M=$M, image $IMAGE"; nvidia-smi -L || true
srun --container-image="$IMAGE" \
     --container-mounts=/path/to:/path/to \
     --export=ALL,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONUNBUFFERED=1,CI_TEXT_CODE_DIR=$WORK \
     python3 $WORK/experiments/exp4_regen.py --engine vllm --model "$MODEL" --M "$M" --n 3000 --jobs-per-chunk 1500 \
       --source $WORK/data/llm/gen_claude-haiku-4-5_n3000.jsonl --out-dir $WORK/out
echo "done: $(wc -l < $WORK/out/regen_qwen3-1.7b_n3000_M${M}.jsonl) draws"
