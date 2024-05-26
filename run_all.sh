#!/bin/bash

# Define the models to run
declare -a models=(
"yentinglin/Llama-3-Taiwan-70B-Instruct"
"yentinglin/Llama-3-Taiwan-8B-Instruct-rc1"
"meta-llama/Meta-Llama-3-70B-Instruct"
"meta-llama/Meta-Llama-3-70B"
"meta-llama/Meta-Llama-3-8B-Instruct"
"meta-llama/Meta-Llama-3-8B"
"Qwen/Qwen1.5-110B-Chat"
"Qwen/Qwen1.5-110B"
"Qwen/Qwen1.5-32B"
"Qwen/Qwen1.5-32B-Chat"
"Qwen/Qwen1.5-72B-Chat"
"Qwen/Qwen1.5-72B"
"Qwen/Qwen1.5-MoE-A2.7B"
"Qwen/Qwen1.5-MoE-A2.7B-Chat"
"Qwen/Qwen1.5-4B"
"Qwen/Qwen1.5-4B-Chat"
"Qwen/Qwen1.5-0.5B"
"Qwen/Qwen1.5-0.5B-Chat"
"Qwen/Qwen1.5-1.8B"
"Qwen/Qwen1.5-7B"
"Qwen/Qwen1.5-14B"
"Qwen/Qwen1.5-14B-Chat"
"deepseek-ai/DeepSeek-V2-Chat"
"01-ai/Yi-1.5-34B"
"01-ai/Yi-1.5-34B-Chat"
"01-ai/Yi-1.5-34B-32K"
"01-ai/Yi-1.5-34B-Chat-16K"
"01-ai/Yi-1.5-9B-32K"
"01-ai/Yi-1.5-9B-Chat-16K"
"01-ai/Yi-1.5-9B"
"01-ai/Yi-1.5-9B-Chat"
"01-ai/Yi-1.5-6B"
"01-ai/Yi-1.5-6B-Chat"
"CohereForAI/c4ai-command-r-plus"
"CohereForAI/c4ai-command-r-v01"
"CohereForAI/aya-23-35B"
"CohereForAI/aya-23-8B"
"mistralai/Mixtral-8x22B-Instruct-v0.1"
"mistralai/Mixtral-8x22B-v0.1"
"mistralai/Mistral-7B-Instruct-v0.3"
"mistralai/Mistral-7B-v0.3"
"mistralai/Mistral-7B-Instruct-v0.2"
"mistralai/Mixtral-8x7B-Instruct-v0.1"
"mistralai/Mixtral-8x7B-v0.1"
"mistralai/Mistral-7B-v0.1"
"MediaTek-Research/Breeze-7B-32k-Instruct-v1_0"
"MediaTek-Research/Breeze-7B-Instruct-v0_1"
"MediaTek-Research/Breeze-7B-Base-v0_1"
"MediaTek-Research/Breeze-7B-Instruct-v1_0"
"MediaTek-Research/Breeze-7B-Base-v1_0"
"INX-TEXT/Bailong-instruct-7B"
"taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
"taide/TAIDE-LX-7B-Chat"
"taide/TAIDE-LX-7B"
"microsoft/Phi-3-mini-4k-instruct"
"apple/OpenELM-3B-Instruct"
)

# SLURM script to be used
SLURM_SCRIPT="harness_eval.slurm"

# Parameters for the script
PARAMS="tmlu,twllm_eval,tw_legal,ccp,pega,tmmluplus,mmlu,pega_mmlu,umtceval"

# Loop through each model and submit a job
for model in "${models[@]}"
do
  echo "Submitting job for $model"
  sbatch $SLURM_SCRIPT $model $PARAMS
done

echo "All jobs submitted"
