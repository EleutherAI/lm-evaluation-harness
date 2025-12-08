#!/bin/bash

source /localhome/tansang/envs/lm-eval/bin/activate

MODELS=(
    "swiss-ai/Apertus-8B-2509"
    "/localhome/tansang/llm/aisingapore/Gemma-SEA-LION-v4-27B"
    # aisingapore/Qwen-SEA-LION-v4-8B-VL
)

TASKS="xnli_en,xnli_th,xnli_vi,xnli_zh,xcopa_en,xcopa_id,xcopa_ta,xcopa_th,xcopa_vi,xcopa_zh,"
TASKS="${TASKS}xcopa_7b-5shot_id_en,xcopa_7b-5shot_ta_en,xcopa_7b-5shot_th_en,xcopa_7b-5shot_vi_en,xcopa_7b-5shot_zh_en,"
TASKS="${TASKS}xcopa_custom_nmt_id_en,xcopa_custom_nmt_ta_en,xcopa_custom_nmt_vi_en,"
TASKS="${TASKS}xcopa_google_id_en,xcopa_google_ta_en,xcopa_google_th_en,xcopa_google_vi_en,xcopa_google_zh_en,"
TASKS="${TASKS}xnli_7b_5shot_th_en,xnli_7b_5shot_vi_en,xnli_7b_5shot_zh_en,"
TASKS="${TASKS}xnli_custom_nmt_vi_en,"
TASKS="${TASKS}xnli_google_th_en,xnli_google_vi_en,xnli_google_zh_en"

for MODEL in "${MODELS[@]}"; do
    BASE_NAME=$(basename "$MODEL")
    OUTPUT_PATH="/localhome/tansang/lm-evaluation-harness/results/${BASE_NAME}"
    mkdir -p "$OUTPUT_PATH"

    lm_eval --model hf \
        --model_args pretrained=$MODEL,parallelize=True \
        --tasks $TASKS \
        --output_path $OUTPUT_PATH \
        --batch_size 16

    echo "Completed evaluation for model: $MODEL"
    sleep 2
done