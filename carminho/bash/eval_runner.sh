#!/bin/bash
#SBATCH --job-name=carminho_eval
#SBATCH --output=./logs/%A-%a.out
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --qos=gpu-short
#SBATCH --partition=a6000
#SSBATCH --array=0

# set -e
export TOKENIZERS_PARALLELISM=false

module load cuda
source ~/mydata/venvs/carminho/bin/activate
echo Interpreter used: `which python`

MODELS=( \
    "carminho/carminho_base_1"
)

# Check if this is an array job
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Running as array job with task ID: $SLURM_ARRAY_TASK_ID"
    MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}
else
    echo "Running as standalone job, checking command line arguments"
    if [ -n "$1" ]; then
        echo "Using model from command line argument: $1"
        MODEL="$1"
    else
        echo "No model specified, using first model from the list"
        MODEL=${MODELS[0]}
    fi
fi

echo "Model: $MODEL"

# model_args="pretrained=${MODEL},tensor_parallel_size=${GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=${model_replicas}"
model_args="pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,max_model_len=2048"
tasks="belebele_por_Latn,global_mmlu_pt"
output_path="/mnt/home/giuseppe0/myscratch/carminho/"
cache_dir="/mnt/home/giuseppe/myscratch/carminho = None/cache/${MODEL//\//__}"

# HF Hub logging arguments
hf_org="hub_results_org=carminho"
hf_details_repo="details_repo_name=test-results"
hf_results_repo="results_repo_name=test-results"
push_results="push_results_to_hub=True"
push_samples="push_samples_to_hub=True"
public_repo="public_repo=False"
# poc="point_of_contact=mail@gmail.com"
# gated="gated=True"

lm_eval --model vllm \
    --model_args $model_args \
    --tasks $tasks \
    --batch_size auto \
    --output_path $output_path \
    --log_samples \
    --use_cache $cache_dir \
    --cache_requests "true" \
    --seed 42 \
    --hf_hub_log_args ${hf_org},${hf_details_repo},${hf_results_repo},${push_results},${push_samples},${public_repo}

echo "Done"