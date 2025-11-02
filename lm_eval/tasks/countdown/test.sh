lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-8B \
    --tasks countdown \
    --device cpu \
    --batch_size 8