#!/bin/bash

batch_size=5
num_fewshot=0

export CUDA_VISIBLE_DEVICES=0,1

model_names=(
  "google/gemma-1.1-7b-it",
  "google/gemma-2-9b-it",
  "google/gemma-2-27b-it",
  "Jacaranda/AfroLlama_V1",
  "LLaMAX/LLaMAX3-8B-Alpaca",
  "meta-llama/Llama-2-7b-chat-hf",
  "meta-llama/Llama-3.1-8B-Instruct",
  "meta-llama/Llama-3.1-70B-Instruct",
  "meta-llama/Meta-Llama-3-8B-Instruct",
  "CohereForAI/aya-101"
)

for model_name in "${model_names[@]}"
do
    echo "Running model: $model_name"
    lm_eval --model hf \
    --model_args pretrained=${model_names},parallelize=true \
    --tasks  afrobench\
    --batch_size ${batch_size} \
    --num_fewshot ${num_fewshot} \
    --verbosity DEBUG \
    --output_path 'path_to_results/' \
    --log_samples
done
