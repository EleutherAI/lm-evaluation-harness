export CUDA_VISIBLE_DEVICES=0,1,6,7
# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen3-32B,parallelize=True \
#     --tasks countdown \
#     --device cuda \
#     --batch_size auto
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-8B \
    --tasks countdown \
    --device cuda \
    --batch_size auto \
    --apply_chat_template
# accelerate launch -m lm_eval --model openai-chat-completions \
#     --model_args model=gpt-5 \
#     --tasks countdown \
#     --device cuda \
#     --batch_size 16 \
#     --apply_chat_template