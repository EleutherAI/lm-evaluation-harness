lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks biology_ds \
    --device cuda:0 \
    --batch_size 8