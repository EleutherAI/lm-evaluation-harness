lm_eval --model hf \
        --model_args pretrained=masakhane/African-ultrachat-alpaca  \
        --tasks afrimmlu_direct_amh,afrimmlu_direct_eng,afrimmlu_direct_ewe,afrimmlu_direct_fra,afrimmlu_direct_hau,afrimmlu_direct_ibo,afrimmlu_direct_kin,afrimmlu_direct_lin,afrimmlu_direct_lug,afrimmlu_direct_orm,afrimmlu_direct_sna,afrimmlu_direct_sot,afrimmlu_direct_twi,afrimmlu_direct_wol,afrimmlu_direct_xho,afrimmlu_direct_yor,afrimmlu_direct_zul   \
        --device cuda:0     \
        --batch_size 1 \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --wandb_args project=afrimmlu
