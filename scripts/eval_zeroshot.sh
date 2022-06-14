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

\time -o ../outputs/time-$1.$task_no_slash.txt -f "%E"  python3 -m main --device cuda --tasks $2 --num_fewshot 0 --model hf-causal --model_args pretrained=$MODEL --no_cache --batch_size $3
