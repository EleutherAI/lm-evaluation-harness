#!/usr/bin/zsh

#SBATCH --job-name=it_eval
#SBATCH --output=logs/%A-%a.out
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=16:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0

export TOKENIZERS_PARALLELISM=false

# source ~/.zshrc
source $FAST/lm_eval/bin/activate

MODELS=( \
    "meta-llama/Meta-Llama-3-8B" \
)

MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}
BATCH_SIZE=1
OUTPUT_DIR=$FAST/tests_calamita

module load cuda

tasks=belebele_ita,veryfIT_enriched
srun accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${MODEL},dtype=bfloat16 \
    --tasks ${tasks} \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --output_path ${OUTPUT_DIR} \
    --use_cache ${OUTPUT_DIR}/cache/${MODEL//\//__} \
    --cache_requests "true"