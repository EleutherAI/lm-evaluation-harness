#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --time=48:00:00
#SBATCH --ntasks=1 
#SBATCH --job-name=lm-harness-dynamic-ifeval
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100GB
#SBATCH --output=/cluster/home/dguidobene/logs/lmh/lmh2.out
#SBATCH --error=/cluster/home/dguidobene/logs/lmh/lmh2.err
#SBATCH --gpus=rtx_3090:1
#SBATCH --tmp=500G

module load--ignore_cache eth_proxy

export HF_HOME=/cluster/scratch/dguidobene

source /cluster/scratch/dguidobene/venvs/lmharness/bin/activate
lm_eval \
  --model vllm \
  --model_args '{"pretrained":"Qwen/Qwen3-1.7B","trust_remote_code":true,"max_model_len":16384,"enable_thinking":true}' \
  --gen_kwargs '{"max_gen_toks":8192,"until":["Input","Input:","<eot_id>","<|im_end|>","###","Question","question","####","Problem","Response"],"temperature":0.6}' \
  --apply_chat_template \
  --tasks dynamic_ifeval \
  --device cuda:0 \
  --batch_size auto