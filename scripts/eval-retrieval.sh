TRITON_PRINT_AUTOTUNING=1 accelerate launch -m lm_eval \
    --model hf \
    --model_args dtype=bfloat16,pretrained=$1 \
    --tasks fda,swde,based_nq_2048,based_triviaqa,squad_completion,based_drop \
    --batch_size 1
