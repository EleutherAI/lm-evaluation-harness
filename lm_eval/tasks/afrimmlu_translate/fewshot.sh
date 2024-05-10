lm_eval --model hf \
        --model_args pretrained=masakhane/African-ultrachat-alpaca  \
        --tasks afrimmlu_amh,afrimmlu_eng,afrimmlu_ewe,afrimmlu_fra,afrimmlu_hau,afrimmlu_ibo,afrimmlu_kin,afrimmlu_lin,afrimmlu_lug,afrimmlu_orm,afrimmlu_sna,afrimmlu_sot,afrimmlu_twi,afrimmlu_wol,afrimmlu_xho,afrimmlu_yor,afrimmlu_zul   \
        --device cuda:0     \
        --batch_size 1 \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --wandb_args project=afrimmlu