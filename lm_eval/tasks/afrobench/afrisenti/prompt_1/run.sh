#!/bin/bash

models=(

  "google/gemma-1.1-7b-it"
  "CohereForAI/aya-101"
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "google/gemma-2-9b-it"
  "bigscience/mt0-xxl"
  "google/gemma-2-27b-it"
  "meta-llama/Meta-Llama-3-70B-Instruct"
)
task=afrisenti_amh_prompt_1,afrisenti_arq_prompt_1,afrisenti_ary_prompt_1,afrisenti_hau_prompt_1,afrisenti_ibo_prompt_1,afrisenti_kin_prompt_1,afrisenti_pcm_prompt_1,afrisenti_por_prompt_1,afrisenti_swa_prompt_1,afrisenti_tir_prompt_1,afrisenti_tso_prompt_1,afrisenti_twi_prompt_1,afrisenti_yor_prompt_1

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 5
  do
    export OUTPUT_DIR=results/$fewshot

    mkdir -p "$OUTPUT_DIR"

    lm_eval --model hf \
            --model_args "pretrained=${model}" \
            --tasks $task\
            --device cuda:0 \
            --batch_size 16 \
            --output_path "$OUTPUT_DIR" \
            --num_fewshot $fewshot \
            --verbosity DEBUG
  done
done
