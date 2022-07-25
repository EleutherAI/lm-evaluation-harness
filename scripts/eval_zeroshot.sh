case $1 in
gptj)
MODEL=EleutherAI/gpt-j-6B
;;

mgpt)
MODEL=sberbank-ai/mGPT
;;

*)
MODEL=$1
esac

task_no_slash=$(basename "$2")

\time -o ../outputs/time-$1.$task_no_slash.txt -f "%E"  python3 -m main \
    --model_api_name 'hf-causal' \
    --model_args pretrained=$MODEL \
    --task_name $2 \
    --num_fewshot 0 \
    --use_cache \
    --device cuda \
    --batch_size $3
