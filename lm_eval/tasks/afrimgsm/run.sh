lm_eval --model hf   \
        --model_args pretrained="google/gemma-7b"  --tasks afrimgsm_en_cot_eng,mgsm_en_cot_en,afrimgsm_native_cot_eng,mgsm_native_cot_en,afrimgsm_direct_eng,mgsm_direct_en,afrimgsm_direct_native_eng  \
        --device cuda:0     \
        --batch_size 1  \
        --verbosity DEBUG \
        --limit 5
