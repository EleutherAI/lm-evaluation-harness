#!/bin/bash

# TASKS=hellaswag,lambada_openai,piqa,arc_easy,arc_challenge,winogrande
TASKS=swde
DEVICE=cuda:0
BATCH_SIZE=32

MODELS=("based-360m" "mamba-360m" "attn-360m")

# for MODEL in "${MODELS[@]}"; do

#   lm_eval \
#       --model based_lm \
#       --model_args checkpoint_name=hazyresearch/$MODEL \
#       --tasks $TASKS \
#       --device $DEVICE \
#       --batch_size $BATCH_SIZE \
#       --limit 100 \
#       --log_samples \
#       --output_path output/$MODEL
    
# done


python launch.py \
  --batch-size 32 \
  -m "hazyresearch/based-360m" \
  -m "hazyresearch/mamba-360m" \
  -m "hazyresearch/attn-360m" \
  -t "swde" \
  -t "hellaswag" \
  -t "lambada_openai" \
  -t "piqa" \
  -t "arc_easy" \
  -t "arc_challenge" \
  -t "winogrande" \
  # --limit 1000 \
  -p