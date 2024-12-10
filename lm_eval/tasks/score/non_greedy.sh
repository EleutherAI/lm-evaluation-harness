#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -m MODEL -t TASK -s SEED -o OUTPUT_DIR"
   echo -e "\t-m huggingface model name"
   echo -e "\t-t task name one of score_non_greedy_robustness_[agieval|mmlu_pro|math]"
   echo -e "\t-s random seed for evaluation [1-5]"
   echo -e "\t-o output directory"
   exit 1 # Exit script after printing help
}

while getopts "m:t:s:" opt
do
   case "$opt" in
      m ) MODEL="$OPTARG" ;;
      t ) TASK="$OPTARG" ;;
      s ) SEED="$OPTARG" ;;
      o ) OUTPUT_DIR="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z "$MODEL" ] | [ -z "$TASK" ] | [ -z "$SEED" ] | [ -z "$OUTPUT_DIR" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

echo "evaluating $MODEL on task $TASK with seed $SEED"
echo "output will be saved in $OUTPUT_DIR"

TENSOR_PARALLEL=8
BATCH_SIZE="auto"

echo "running evaluation on vllm with tensor parallelism $TENSOR_PARALLEL"

lm_eval --model vllm \\
 --model_args pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$TENSOR_PARALLEL,gpu_memory_utilization=0.9,\\
 max_model_len=4096,data_parallel_size=1,disable_custom_all_reduce=True,enforce_eager=False,seed=$SEED\\
 --apply_chat_template \\
 --tasks $TASKS \\
 --batch_size $BATCH_SIZE \\
 --log_samples \\
 --output_path $OUTPUT_DIR \\
