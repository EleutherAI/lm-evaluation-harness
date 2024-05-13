#!/bin/bash

models=(
  "gpt-3.5-turbo"
  "gpt-4-0125-preview"
)
task=afrimgsm_direct_eng,afrimgsm_direct_fra,afrimgsm_direct_swa #afrimgsm_direct_ewe,afrimgsm_direct_fra,afrimgsm_direct_hau,afrimgsm_direct_ibo,afrimgsm_direct_kin,afrimgsm_direct_lin,afrimgsm_direct_lug,afrimgsm_direct_orm,afrimgsm_direct_sna,afrimgsm_direct_sot,afrimgsm_direct_swa,afrimgsm_direct_twi,afrimgsm_direct_wol,afrimgsm_direct_xho,afrimgsm_direct_yor,afrimgsm_direct_zul

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 2 4 6 8
  do
    export OUTPUT_DIR=results/$fewshot

    mkdir -p "$OUTPUT_DIR"

    lm_eval --model openai-chat-completions \
            --model_args model="${model}" \
            --tasks $task \
            --output_path "$OUTPUT_DIR" \
            --num_fewshot $fewshot \
            --verbosity DEBUG
  done
done