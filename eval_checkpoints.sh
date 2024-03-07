#!/bin/bash

# Check if two arguments are provided
if [ $# -ne 4 ]; then
  echo "Usage: $0 <models_directory> <csv_output> <lang> <model>"
  exit 1
fi

# Directory containing model folders
models_directory="$1"

# Check if the directory exists
if [ ! -d "$models_directory" ]; then
  echo "Error: Directory $models_directory does not exist."
  exit 1
fi

# Concatenate models_directory with csv_output
csv_output="$models_directory/$2"

# Function to extract model names from res.csv
extract_model_names() {
  if [ -f "$1" ]; then
    awk -F',' 'NR>1 {print $1}' "$1"
  else
    echo ""
  fi
}

# Get existing model names from res.csv or an empty string if it doesn't exist
existing_models=$(extract_model_names "$csv_output")

# Create a directory to store output files
output_dir="output_tables"
mkdir -p "$output_dir"

# Populate model_args_list with subdirectories of the provided directory
model_args_list=()
for model_dir in "$models_directory"/*/; do
  model_name="$models_directory${model_dir#$models_directory}"
  # Check if the model has already been evaluated
  if ! grep -q "^$model_name$" <<< "$existing_models"; then
    model_args_list+=("pretrained=$model_name")
  else
    echo "Model $model_name already evaluated. Skipping..."
  fi
done

# Prepare task list
lang=$3
if [ "$lang" = "ru" ]; then
  tasks="winogrande_ru,arc_challenge_ru,hellaswag_ru,mmlu_ru,gsm8k_ru,truthfulqa_mc2_ru"
else
  tasks="winogrande,arc_challenge,hellaswag,mmlu,gsm8k,truthfulqa_mc2"
fi

# Prepare model name
model="hf"
if [ "$4" = "mamba" ]; then
  model="mamba_ssm"
fi

# Iterate over the list and run the command for new models
for model_args_value in "${model_args_list[@]}"; do
  command="accelerate launch -m  lm_eval --model $model --tasks $tasks --batch_size 4 --model_args $model_args_value --output_csv $csv_output"
  output_file="$output_dir/output_${model_args_value//\//_}.txt"  # Replace "/" with "_"
  echo "Running command: $command"
  $command > "$output_file"  # Redirect output to file
done

# Print the contents of the output files
for output_file in "$output_dir"/*.txt; do
  echo -e "\nContents of $output_file:"
  cat "$output_file"
done

# Clean-up
rm -r "$output_dir"
