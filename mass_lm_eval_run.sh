#!/bin/bash

MODELS=(
    # "/scratch/e1583535/llm/nus-olmo/mixed-n10B"
    # "/scratch/e1583535/llm/nus-olmo/para-first-n10B"
    # "/scratch/e1583535/llm/nus-olmo/para-last-n10B-rerun"
    # "/scratch/e1583535/llm/nus-olmo/multi-uniform-n10B-SEA-7.5_replay-2.5-checkpoints/step4770-unsharded-hf-multi-uniform"
    # "/scratch/e1583535/llm/nus-olmo/multilingual-n10B-7.5-replay-2.5-checkpoints/step4770-unsharded-hf-multilingual"
    # "/scratch/e1583535/llm/nus-olmo/para-replay-n10B"
    # "/scratch/e1583535/llm/nus-olmo/para-only-34B8"
    # "/scratch/e1583535/llm/nus-olmo/multilingual-uniform-7B_n34.8-26_replay-8.7-checkpoints/step8290-unsharded-hf-multilingual-uniform-34.7B"
    # "/scratch/e1583535/llm/nus-olmo/multilingual-7B_n34.8-26_replay-8.7-checkpoints/step8290-unsharded-hf-multilingual-7B-34.7B"
    # "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step8290-unsharded-hf-para-only-7B-34.7B"
    # "SeaLLMs/SeaLLMs-v3-1.5B"
    # "sail/Sailor2-L-1B"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "aisingapore/Llama-SEA-LION-v3.5-8B-R"
    # "sail/Sailor2-8B"
    # "SeaLLMs/SeaLLMs-v3-7B"
    # "allenai/OLMo-2-1124-7B"
    # "aisingapore/Gemma-SEA-LION-v4-27B"
    # "aisingapore/Qwen-SEA-LION-v4-32B-IT"
    # "/scratch/e1583535/llm/sail/Sailor2-20B"
    # "/scratch/e1583535/llm/openseal-sft/openseal-sailor2ds-stage1"
    # "/scratch/e1583535/llm/openseal-sft/openseal-sailor2ds-stage2"
    # "/scratch/e1583535/llm/openseal-dpo/openseal_dpo_sailor2_stage1_8gpu"
)

# NOTE: space-separated here (no commas!)
# TASKS="xnli_en xnli_th xnli_vi xnli_zh xcopa_en xcopa_id xcopa_ta xcopa_th xcopa_vi xcopa_zh xcopa_7b-5shot_id_en xcopa_7b-5shot_ta_en xcopa_7b-5shot_th_en xcopa_7b-5shot_vi_en xcopa_7b-5shot_zh_en xcopa_google_id_en xcopa_google_ta_en xcopa_google_th_en xcopa_google_vi_en xcopa_google_zh_en xnli_7b_5shot_th_en xnli_7b_5shot_vi_en xnli_7b_5shot_zh_en xnli_google_th_en xnli_google_vi_en xnli_google_zh_en paws_en paws_zh"
TASKS="xnli_en xnli_th xnli_vi xnli_zh xcopa_en xcopa_id xcopa_ta xcopa_th xcopa_vi xcopa_zh paws_en paws_zh copal_id_colloquial copal_id_standard kalahi_tl"

TYPE="xnli-xcopa-pawsx-multilingual-260105"
BASE_LOG_DIR="/scratch/e1583535/multilingual-llm-project/logs/eval/lm-evaluation-harness/$TYPE"

mkdir -p "$BASE_LOG_DIR"

for MODEL in "${MODELS[@]}"; do
    BASE_NAME=$(basename "$MODEL")
    OUTPUT_PATH="/scratch/e1583535/results/lm_evaluation_harness/${TYPE}/$BASE_NAME"
    mkdir -p "$OUTPUT_PATH"

    qsub -v MODEL="$MODEL",TASKS="$TASKS",OUTPUT_PATH="$OUTPUT_PATH",BASE_LOG_DIR="$BASE_LOG_DIR" mass_lm_eval_run.pbs

    echo "Submitted job for model: $MODEL"
    sleep 2
done
