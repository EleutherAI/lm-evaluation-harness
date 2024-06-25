#!/bin/bash

# List of different model_args values
model_args_list=(
   "pretrained=deepvk/llama-1.5b-sft"
   "pretrained=deepvk/llama-3b-sft"
)

# Create a directory to store output files
output_dir="output_tables"
mkdir -p $output_dir

# tasks="mmlu,mmlu_continuation,mmlu_generative,mmlu_full_choice"
tasks="winogrande,arc_challenge,hellaswag,mmlu,gsm8k,truthfulqa_mc2"


# Iterate over the list and run the command
for model_name in "${model_names[@]}"; do
  command="accelerate launch -m lm_eval --model hf --tasks $tasks --batch_size auto --model_args pretrained=$model_name"
  output_file="$output_dir/output_${model_name//\//_}.txt"  # Replace "/" with "_"
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
