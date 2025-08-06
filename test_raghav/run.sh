export OPENAI_API_KEY=sk-rc-R-vJqSca2wRZBX5qBAGaqg

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2-0.5B \
    --tasks multijail \
    --batch_size 8 \
    --device cuda:0 \
    --limit 10
    