#!/bin/bash

# TASKS=hellaswag,lambada_openai,piqa,arc_easy,arc_challenge,winogrande
TASKS=swde
DEVICE=cuda:0
BATCH_SIZE=32


declare -A MODELS
# MODELS[based]=hazyresearch/based-1.3b
# MODELS[mamba]=hazyresearch/mamba-1.3b
MODELS[transformer]=hazyresearch/transformer-pp-1.3b

for MODEL in "${!MODELS[@]}"; do
  CHECKPOINT_NAME=${MODELS[$MODEL]}
  python run_harness.py --model based_hf \
      --model_args checkpoint_name=$CHECKPOINT_NAME,model=$MODEL \
      --tasks $TASKS \
      --device $DEVICE \
      --batch_size $BATCH_SIZE \
      --limit 100 \
      --log_samples \
      --output_path output/$MODEL
    
done
