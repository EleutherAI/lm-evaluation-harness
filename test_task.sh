#!/bin/bash

#SBATCH --job-name="eqbench_test"
#SBATCH -D .
#SBATCH --output=/gpfs/projects/bsc88/mlops-lm-evaluation-harness/eqbench_pr/logs/%j.log
#SBATCH --partition=acc
#SBATCH --qos acc_debug
#SBATCH --account bsc88
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -t 02:00:00

grep SBATCH "$0" | head -n -1 # log the SBATCH options

cd /gpfs/projects/bsc88/mlops-lm-evaluation-harness/eqbench_pr/

export HF_HUB_OFFLINE=1

export HF_HOME="/gpfs/projects/bsc88/hf-home"
export TRUST_REMOTE_CODE=true
export HF_DATASETS_TRUST_REMOTE_CODE=true

# PAULA
model_name=FLOR-760M
task=eqbench_ca,eqbench_es
apply_chat_template=False
fewshot=5

echo "Model: ${model_name}"
echo "Task: ${task}"
echo "Apply chat template: ${apply_chat_template}"

timestamp=$(date +%Y-%m-%dT%H%M-%S)

model_path=/gpfs/projects/bsc88/hf-models/${model_name}

echo "Model path: $model_path"

output_path=/gpfs/projects/bsc88/mlops-lm-evaluation-harness/eqbench_pr/results/${model_name}/results_${model_name}:${timestamp}.json
echo "Output path: ${output_path}"

common_args="--tasks ${task} \
    --num_fewshot ${fewshot} \
    --batch_size auto \
    --output_path ${output_path} \
    --seed 1234 \
    --log_samples \
    --verbosity DEBUG \
    --model_args pretrained=${model_path},trust_remote_code=True \
    --limit 10" # PAULA: sin limite para testear la task, con limite para testear el grupo :)

if [[ "${apply_chat_template}" == "True" ]]; then
    echo "Applying chat template"
    common_args="$common_args --apply_chat_template --fewshot_as_multiturn"
fi

launch_command="python -m lm_eval ${common_args}" # PAULA: este es el comando que llama el harness!
echo "Command: ${launch_command}"

module load singularity

export SINGULARITY_IMAGE_LLAMA_TAG="galtea-llmops-sentence-transformers_3.3.1.sif"
export SINGULARITY_IMAGES_DIR="/gpfs/projects/bsc88/singularity-images"

export HF_DATASETS_OFFLINE="1"
export HF_HOME="/gpfs/projects/bsc88/hf-home"
export LD_LIBRARY_PATH=/apps/ACC/CUDA/12.3/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
export SINGULARITY_CACHEDIR=./cache_singularity
export SINGULARITY_TMPDIR=./cache_singularity
export NUMBA_CACHE_DIR=./cache_numba
export TORCHDYNAMO_SUPPRESS_ERRORS=True
export MLCONFIGDIR=/gpfs/scratch/bsc88/bsc088532/.cache/matplotlib
export NUMEXPR_MAX_THREADS=64
export VLLM_CONFIG_ROOT=$TMPDIR
export VLLM_CACHE_ROOT=$TMPDIR

singularity run --nv --no-home -B $TMPDIR:/tmp "${SINGULARITY_IMAGES_DIR}"/"${SINGULARITY_IMAGE_LLAMA_TAG}" bash -c "${launch_command}"
