#!/bin/bash

# List of different model_args values
model_args_list=(
   "pretrained=deepvk/llama-1.5b-sft"
   "pretrained=deepvk/llama-3b-sft"
)

# Iterate over the list and run the command
for model_args_value in "${model_args_list[@]}"; do
  command="accelerate launch -m lm_eval --model hf --tasks winogrande_ru,arc_challenge_ru,hellaswag_ru,mmlu_ru,gsm8k_ru,truthfulqa_mc2_ru --batch_size 4 --model_args $model_args_value"
  echo "Running command: $command"
  $command
done
