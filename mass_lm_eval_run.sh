#!/bin/bash

MODELS=(
    "/scratch/e1583535/llm/nus-olmo/mixed-n10B"
    "/scratch/e1583535/llm/nus-olmo/para-first-n10B"
    "/scratch/e1583535/llm/nus-olmo/para-last-n10B-rerun"
    "/scratch/e1583535/llm/nus-olmo/para-replay-n10B"
    "/scratch/e1583535/llm/nus-olmo/para-only-34B8"
    "/scratch/e1583535/llm/nus-olmo/para-last-100B-checkpoints/step42931-unsharded-hf"
    "/scratch/e1583535/llm/nus-olmo/para-last-100B"
    "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step2385-unsharded-hf"
    "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step6201-unsharded-hf"
    "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step8290-unsharded-hf"
    "SeaLLMs/SeaLLMs-v3-1.5B"
    "sail/Sailor2-L-1B"
    "aisingapore/Llama-SEA-LION-v3.5-8B-R"
)

# NOTE: space-separated here (no commas!)
TASKS="xcopa_google_id_en xcopa_google_ta_en xcopa_google_th_en xcopa_google_vi_en xcopa_google_zh_en"

TYPE="xcopa_google_translate"
BASE_LOG_DIR="/scratch/e1583535/multiLingual-llm-project/logs/eval/lm-evaluation-harness/$TYPE"

mkdir -p "$BASE_LOG_DIR"

for MODEL in "${MODELS[@]}"; do
    BASE_NAME=$(basename "$MODEL")
    OUTPUT_PATH="/scratch/e1583535/results/lm_evaluation_harness/${TYPE}/$BASE_NAME"
    mkdir -p "$OUTPUT_PATH"

    qsub -v MODEL="$MODEL",TASKS="$TASKS",OUTPUT_PATH="$OUTPUT_PATH",BASE_LOG_DIR="$BASE_LOG_DIR" mass_lm_eval_run.pbs

    echo "Submitted job for model: $MODEL"
    sleep 2
done
