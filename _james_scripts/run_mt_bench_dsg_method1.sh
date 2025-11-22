#!/bin/bash
#SBATCH --job-name=mt-bench
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/jwedgwoo/FastChat/fastchat/llm_judge
#SBATCH --output=/home/jwedgwoo/FastChat/logs/%x-%j.log
#SBATCH --error=/home/jwedgwoo/FastChat/logs/%x-%j.log

set -euo pipefail

# activate venv
source /home/jwedgwoo/FastChat/.venv/bin/activate

echo "Evaluating model answers ..."
export OPENAI_API_KEY="$(cat /home/jwedgwoo/FastChat/openai_api_key.txt)"
stdbuf -oL -eL python gen_judgment.py --model-list gemma-SFT-method1-conditional gemma-SFT-method1-conditional-antirepetition 2>&1 | \
    tee -a "/home/jwedgwoo/FastChat/logs/mt-bench-$SLURM_JOB_ID.out"