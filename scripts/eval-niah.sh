TRITON_PRINT_AUTOTUNING=1 accelerate launch -m lm_eval \
    --model hf \
    --model_args dtype=bfloat16,pretrained=$1,max_length=32768 \
    --tasks niah_single_1,niah_single_2,niah_single_3 \
    --metadata='{"max_seq_lengths":[1024,2048,4096,8192,16384]}' \
    --batch_size 1
