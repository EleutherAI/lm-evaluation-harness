TRITON_PRINT_AUTOTUNING=1 accelerate launch -m lm_eval \
    --model hf \
    --model_args dtype=bfloat16,pretrained=$1 \
    --tasks based_drop \
    --limit 100 \
    --metadata '{"max_seq_lengths":[2048]}' \
    --batch_size 1
