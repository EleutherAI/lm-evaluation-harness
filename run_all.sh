#!/bin/bash

# Define the models to run
declare -a models=(
"yentinglin/Llama-3-Taiwan-70B-Instruct"
"yentinglin/Taiwan-Llama-3-70B-Cooldown"
"yentinglin/Taiwan-Llama-3-70B"
"yentinglin/Taiwan-Llama-3-8B-Instruct"
"yentinglin/Taiwan-Llama-3-8B-Cooldown"
"yentinglin/Taiwan-Llama-3-8B"
"meta-llama/Meta-Llama-3-70B-Instruct"
"meta-llama/Meta-Llama-3-70B"
"meta-llama/Meta-Llama-3-8B-Instruct"
"meta-llama/Meta-Llama-3-8B"
"Qwen/Qwen1.5-110B-Chat"
"Qwen/Qwen1.5-72B-Chat"
"deepseek-ai/DeepSeek-V2-Chat"
"01-ai/Yi-34B-Chat"
"CohereForAI/c4ai-command-r-plus"
"mistralai/Mixtral-8x22B-Instruct-v0.1"
"MediaTek-Research/Breeze-7B-Instruct-v1_0"
"MediaTek-Research/Breeze-7B-Base-v1_0"
"taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
"taide/TAIDE-LX-7B-Chat"
"taide/TAIDE-LX-7B"
"microsoft/Phi-3-mini-4k-instruct"
"apple/OpenELM-3B-Instruct"
)

# SLURM script to be used
SLURM_SCRIPT="harness_eval.slurm"

# Parameters for the script
PARAMS="tmlu,twllm_eval,tw_legal,ccp,pega,tmmluplus"

# Loop through each model and submit a job
for model in "${models[@]}"
do
  echo "Submitting job for $model"
  sbatch $SLURM_SCRIPT $model $PARAMS
done

echo "All jobs submitted"
