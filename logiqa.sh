#!/bin/bash

# Array of model paths
model_paths=(
  "bigscience/bloom-7b1"
  "bigscience/bloom-3b"
  "bigscience/bloom"
)

# Array of output paths
output_paths=(
  "bloom7"
  "bloom3"
  "bloomfull"
)

# Iterate over the arrays and run the script for each model and output path
for ((i=0; i<${#model_paths[@]}; i++)); do
  accelerate launch your_script.py --model hf --model_args pretrained="${model_paths[i]}" --tasks logiqa --batch_size 32 --output_path "${output_paths[i]}"
done
