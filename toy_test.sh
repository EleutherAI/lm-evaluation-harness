#!/bin/bash

model=#Insert the modle you want here
few_shot=0
tensor_parallelism=False
num_samples=2
output_dir=results/$(basename ${model})/${few_shot}-shot/results:$(basename ${model}):${dataset}:-shot.json
dataset=proof-pile

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lm_eval --model hf \
    --model_args pretrained=$model,trust_remote_code=True \
    --tasks ${dataset} \
    --num_fewshot $few_shot \
    --batch_size 1 \
    --output_path $output_dir \
    --log_samples \
    --seed 1234 \
    --limit $num_samples

