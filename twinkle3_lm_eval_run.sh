#!/bin/bash

source /localhome/tansang/envs/lm-eval/bin/activate

MODELS=(
    # "swiss-ai/Apertus-8B-2509"
    # "/localhome/tansang/llm/sail/Sailor2-20B"
    "/localhome/tansang/llm/openseal-sft/openseal-sailor2ds-stage1"
)

### Sample combined tasks ###
# TASKS="xnli_en,xnli_th,xnli_vi,xnli_zh,xcopa_en,xcopa_id,xcopa_ta,xcopa_th,xcopa_vi,xcopa_zh,"
# TASKS="${TASKS}xcopa_7b-5shot_id_en,xcopa_7b-5shot_ta_en,xcopa_7b-5shot_th_en,xcopa_7b-5shot_vi_en,xcopa_7b-5shot_zh_en,"
# TASKS="${TASKS}xcopa_custom_nmt_id_en,xcopa_custom_nmt_ta_en,xcopa_custom_nmt_vi_en,"
# TASKS="${TASKS}xcopa_google_id_en,xcopa_google_ta_en,xcopa_google_th_en,xcopa_google_vi_en,xcopa_google_zh_en,"
# TASKS="${TASKS}xnli_7b_5shot_th_en,xnli_7b_5shot_vi_en,xnli_7b_5shot_zh_en,"
# TASKS="${TASKS}xnli_custom_nmt_vi_en,"
# TASKS="${TASKS}xnli_google_th_en,xnli_google_vi_en,xnli_google_zh_en"

### Simple tasks ###
TASKS="xnli_en,xnli_th,xnli_vi,xnli_zh,xcopa_en,xcopa_id,xcopa_ta,xcopa_th,xcopa_vi,xcopa_zh,paws_en,paws_zh,copal_id_colloquial,copal_id_standard,kalahi_tl"

for MODEL in "${MODELS[@]}"; do
    BASE_NAME=$(basename "$MODEL")
    OUTPUT_PATH="/localhome/tansang/lm-evaluation-harness/results/${BASE_NAME}"
    mkdir -p "$OUTPUT_PATH"

    lm_eval --model hf \
        --model_args pretrained=$MODEL,parallelize=True \
        --tasks $TASKS \
        --log_samples \
        --output_path $OUTPUT_PATH \
        --batch_size 16

    echo "Completed evaluation for model: $MODEL"
    sleep 2
done