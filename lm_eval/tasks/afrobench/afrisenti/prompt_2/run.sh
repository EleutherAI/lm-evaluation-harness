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

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 5
  do
    export OUTPUT_DIR=./results/$fewshot

    mkdir -p "$OUTPUT_DIR"

    lm_eval --model hf \
            --model_args "pretrained=${model},parallelize: true" \
            --tasks afribench\
            --batch_size 256 \
            --output_path "$OUTPUT_DIR" \
            --num_fewshot $fewshot \
            --verbosity DEBUG \
            --limit 2
  done
done
