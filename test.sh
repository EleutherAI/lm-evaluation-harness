lm_eval --model vllm \
    --model_args pretrained=/home/qinbowen/just_malou/model/Qwen/CodeQwen1.5-7B-Chat,dtype=bfloat16 \
    --tasks humaneval_greedy \
    --batch_size 1 \
    --log_samples \
    --output_path results \
    --device cuda:1