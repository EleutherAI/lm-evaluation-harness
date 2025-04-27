CKPT_DIR=$1
OUTPUT_DIR=${2:-"results"}

mkdir -p ${OUTPUT_DIR}

accelerate launch --multi_gpu --num_processes 8 -m lm_eval \
    --model hf \
    --model_args "pretrained=${CKPT_DIR},parallelize=True,dtype=bfloat16,attn_implementation=flash_attention_2" \
    --output_path ${OUTPUT_DIR} \
    --apply_chat_template \
    --batch_size 20 \
    --tasks milu \
    --log_samples