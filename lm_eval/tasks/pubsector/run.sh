#!/usr/bin/zsh

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --job-name=pub_sector_eval
#SBATCH --output=/home/dku84540/logs/pub_sector_eval/slurm_logs/%A_%a.out
#SBATCH --error=/home/dku84540/logs/pub_sector_eval/slurm_logs/%A_%a.out

#SBATCH --account=rwth1961

nvidia-smi

 

module --force purge
module load GCCcore/13.2.0 Python/3.11.5 CUDA
source /home/dku84540/code/lm-evaluation-harness/.venv/bin/activate
export HF_HOME="/rwthfs/cluster.rz.rwth-aachen.de/hpcwork/rwth1961/gov_teuken/cache"
# env LOGLEVEL=DEBUG

# Legal CE BGH Task
lm_eval --model hf \
    --model_args pretrained=mistralai/Ministral-3-3B-Base-2512,trust_remote_code=True \
    --tasks legal_ce_bgh \
    --device cuda:0 \
    --batch_size 32 \
    --num_fewshot 5 \
    --apply_chat_template \
    --output  /home/dku84540/code/lm-evaluation-harness/results/ce_bgh_top_10_percent_eval_results.json

# Public Sector QA Task
#  lm_eval --model hf \
#     --model_args pretrained=openGPT-X/Teuken-7B-instruct-v0.4,trust_remote_code=True \
#     --tasks pubsector \
#     --device cuda:0 \
#     --batch_size 32 \
#     --num_fewshot 5 \
#     --apply_chat_template \
#     --output /home/dku84540/code/lm-evaluation-harness/results/pubsector_eval_results_Teuken-v0.4.json


# TruthfulQA Generation Task
# lm_eval --model hf \
#     --model_args pretrained=openGPT-X/Teuken-7B-instruct-v0.6,trust_remote_code=True \
#     --tasks truthfulqa_gen \
#     --device cuda:0 \
#     --batch_size 32 \
#     --num_fewshot 5 \
#     --apply_chat_template \
#     --output /home/anbh299g/code/lm-evaluation-harness/results/truthfulqa_gen_eval_results_Teuken-v0.6.json