#!/bin/bash

models=(
  "masakhane/African-ultrachat-alpaca"
  "masakhane/zephyr-7b-gemma-sft-african-alpaca"
  "masakhane/zephyr-7b-gemma-sft-african-ultrachat-5k"
  "google/flan-t5-xxl"
  "bigscience/mt0-xxl-mt"
  "CohereForAI/aya-101"
  "bigscience/bloomz-7b1-mt"
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "meta-llama/Meta-Llama-3-70B-Instruct"
  "google/gemma-1.1-7b-it"
  "RWKV/v5-EagleX-v2-7B-HF"
  "RWKV/rwkv-6-world-7b"
)

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for task in afrimgsm_direct_eng afrimgsm_direct_amh afrimgsm_direct_ibo afrimgsm_direct_fra afrimgsm_direct_sna afrimgsm_direct_lin afrimgsm_direct_wol afrimgsm_direct_ewe afrimgsm_direct_lug afrimgsm_direct_xho afrimgsm_direct_kin afrimgsm_direct_twi afrimgsm_direct_zul afrimgsm_direct_orm afrimgsm_direct_yor afrimgsm_direct_hau afrimgsm_direct_sot afrimgsm_direct_swa
  do
    export OUTPUT_DIR=results/${model##*/}/${task}

    mkdir -p "$OUTPUT_DIR"

    lm_eval --model hf \
            --model_args "pretrained=${model}" \
            --tasks task\
            --device cuda:0     \
            --batch_size 16 \
            --num_fewshot 0 \
            --verbosity DEBUG
  done
done