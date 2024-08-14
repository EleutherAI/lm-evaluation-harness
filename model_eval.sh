#!/bin/bash

# Install required package
pip3 install antlr4-python3-runtime==4.11 immutabledict langdetect accelerate

MODEL_PATHS=( # This can be a local directory OR a huggingface repo, put as many as you want to test, it will run them sequentially.
 TinyLlama/TinyLlama-1.1B-Chat-v1.0
)

tasks=(
"leaderboard_alghafa_light"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME=$(basename "$MODEL_PATH")
  MODEL_DIR="./results/$MODEL_NAME"
  mkdir -p "$MODEL_DIR"
  
  MODEL_ARGS="trust_remote_code=True,pretrained=$MODEL_PATH,dtype=float16"
  
  for TASK in "${tasks[@]}"; do
    lm_eval --model hf --model_args "$MODEL_ARGS" --tasks="$TASK" --batch_size 4  --output_path "$MODEL_DIR/$TASK"
  done
done

