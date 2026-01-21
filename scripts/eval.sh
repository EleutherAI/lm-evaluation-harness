MODEL=/u/shawntan/proj/mayank/checkpoints/hybrid-attn-mamba2-2.5b/unsharded_model

TRITON_PRINT_AUTOTUNING=1 accelerate launch -m lm_eval \
    --model hf \
    --model_args dtype=bfloat16,pretrained=$MODEL \
    --tasks wikitext,openbookqa,piqa,sciq,arc_easy,arc_challenge,boolq,copa,hellaswag,winogrande,race,lambada_openai \
    --batch_size auto
