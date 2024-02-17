#!/bin/bash

# List of different model_args values
model_args_list=(
   "pretrained=deepvk/llama-1.5b-sft"
   "pretrained=deepvk/llama-3b-sft"
)

# Create a directory to store output files
output_dir="output_tables"
mkdir -p $output_dir

# Iterate over the list and run the command
for model_args_value in "${model_args_list[@]}"; do
  command="accelerate launch -m lm_eval --model hf --tasks winogrande_ru,arc_challenge_ru,hellaswag_ru,mmlu_ru,gsm8k_ru,truthfulqa_mc2_ru --batch_size 4 --model_args $model_args_value"
  output_file="$output_dir/output_${model_args_value//\//_}.txt"  # Replace "/" with "_"
  echo "Running command: $command"
  $command > $output_file  # Redirect output to file
done

# Print the contents of the output files
for output_file in $output_dir/*.txt; do
  echo -e "\nContents of $output_file:"
  cat $output_file
done

# Clean-up
rm -r $output_dir
