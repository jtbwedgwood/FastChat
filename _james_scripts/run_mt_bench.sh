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

# pull in remote models from s3
pip install awscli

s3_path_sft="s3://james-cmu-storage/experiments/gemma-SFT-v2/final_checkpoint"
# s3_path_scit="s3://james-cmu-storage/experiments/gemma-SCIT-final"
local_path_sft="/data/user_data/jwedgwoo/gemma-SFT-v2/final_checkpoint"
# local_path_scit="/data/user_data/jwedgwoo/gemma-SCIT-final"

echo "Syncing SFT model from S3..."
aws s3 sync "$s3_path_sft" "$local_path_sft" --delete
# echo "Syncing SCIT model from S3..."
# aws s3 sync "$s3_path_scit" "$local_path_scit" --delete

# also need to bring in gemma-2-2b tokenizer
BASE_MODEL="google/gemma-2-2b"
export HF_TOKEN=$(cat /home/jwedgwoo/RAHF/huggingface_token.txt)
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME="/data/user_data/jwedgwoo/hf_cache"
mkdir -p $HF_HOME
for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json; do
    if [ ! -f "$local_path_sft/$f" ]; then
        echo "Fetching $f..."
        huggingface-cli download "$BASE_MODEL" "$f" --local-dir "$local_path_sft"
    fi
    # if [ ! -f "$local_path_scit/$f" ]; then
    #     echo "Fetching $f..."
    #     huggingface-cli download "$BASE_MODEL" "$f" --local-dir "$local_path_scit"
    # fi
done

echo "Generating answers with SFT model ..."
stdbuf -oL -eL python gen_model_answer.py --model-path "$local_path_sft" --model-id "gemma-SFT-v2" 2>&1 | \
    tee "/home/jwedgwoo/FastChat/logs/mt-bench-$SLURM_JOB_ID.out"
# echo "Generating answers with SCIT model ..."
# stdbuf -oL -eL python gen_model_answer.py --model-path "$local_path_scit" --model-id "gemma-SCIT-final" 2>&1 | \
#     tee "/home/jwedgwoo/FastChat/logs/mt-bench-$SLURM_JOB_ID.out"

echo "Evaluating model answers ..."
export OPENAI_API_KEY="$(cat /home/jwedgwoo/FastChat/openai_api_key.txt)"
stdbuf -oL -eL python gen_judgment.py --model-list gemma-SFT-v2 2>&1 | \
    tee -a "/home/jwedgwoo/FastChat/logs/mt-bench-$SLURM_JOB_ID.out"