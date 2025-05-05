lm_eval --model hf \
        --model_args pretrained=masakhane/African-ultrachat-alpaca  \
        --tasks afrimmlu_direct_amh,afrimmlu_direct_eng,afrimmlu_direct_ewe,afrimmlu_direct_fra,afrimmlu_direct_hau,afrimmlu_direct_ibo,afrimmlu_direct_kin,afrimmlu_direct_lin,afrimmlu_direct_lug,afrimmlu_direct_orm,afrimmlu_direct_sna,afrimmlu_direct_sot,afrimmlu_direct_twi,afrimmlu_direct_wol,afrimmlu_direct_xho,afrimmlu_direct_yor,afrimmlu_direct_zul   \
        --device cuda:0     \
        --batch_size 1 \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --wandb_args project=afrimmlu


lm_eval --model hf \
        --model_args pretrained=bigscience/mt0-small,parallelize=true \
        --tasks afrisenti_amh_prompt_1,afrisenti_arq_prompt_1,afrisenti_ary_prompt_1,afrisenti_hau_prompt_1,afrisenti_ibo_prompt_1,afrisenti_kin_prompt_1,afrisenti_orm_prompt_1,afrisenti_pcm_prompt_1,afrisenti_por_prompt_1,afrisenti_swa_prompt_1,afrisenti_tir_prompt_1,afrisenti_tso_prompt_1,afrisenti_twi_prompt_1,afrisenti_yor_prompt_1\
        --device cuda:0     \
        --batch_size 1 \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --limit 5


lm_eval --model hf \
        --model_args pretrained=bigscience/mt0-xxl,parallelize=true  \
        --tasks afrisenti_amh_prompt_1,afrisenti_arq_prompt_1,afrisenti_ary_prompt_1,afrisenti_hau_prompt_1,afrisenti_ibo_prompt_1,afrisenti_kin_prompt_1,afrisenti_orm_prompt_1,afrisenti_pcm_prompt_1,afrisenti_por_prompt_1,afrisenti_swa_prompt_1,afrisenti_tir_prompt_1,afrisenti_tso_prompt_1,afrisenti_twi_prompt_1,afrisenti_yor_prompt_1\
        --batch_size 128 \
        --num_fewshot 0 \
        --verbosity DEBUG

lm_eval --model hf \
        --model_args pretrained=google/gemma-2-27b-it,parallelize=true,trust_remote_code=True \
        --tasks afriqa_wol_prompt_2\
        --batch_size 1 \
        --device 'cuda' \
        --num_fewshot 5 \
        --verbosity DEBUG \
        --output_path './afriqa_results/' \
        --log_samples

lm_eval --model vllm \
        --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,tensor_parallel_size=2,dtype='auto',gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks masakhapos_pcm_prompt_1,masakhapos_pcm_prompt_2,masakhapos_pcm_prompt_3,masakhapos_pcm_prompt_4,masakhapos_pcm_prompt_5 \
        --batch_size 'auto' \
        --device 'cuda' \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --limit 2


lm_eval --model vllm \
        --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,tensor_parallel_size=2,dtype='auto',gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks masakhapos_pcm_prompt_1,masakhapos_pcm_prompt_2,masakhapos_pcm_prompt_3,masakhapos_bam_prompt_2,masakhapos_bbj_prompt_3 \
        --batch_size 'auto' \
        --device 'cuda' \
        --num_fewshot 0 \
        --verbosity DEBUG

lm_eval --model vllm \
        --model_args pretrained=google/gemma-1.1-7b-it,tensor_parallel_size=2,dtype='auto',gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks masakhaner_pcm_prompt_1\
        --batch_size 'auto' \
        --device 'cuda' \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --limit 5

lm_eval --model vllm \
        --model_args pretrained=google/gemma-2-9b-it,tensor_parallel_size=2,dtype='auto',gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks masakhaner_pcm_prompt_1,masakhaner_pcm_prompt_2,masakhaner_pcm_prompt_3,masakhaner_pcm_prompt_4,masakhaner_pcm_prompt_5\
        --batch_size 'auto' \
        --device 'cuda' \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --limit 5

lm_eval --model vllm \
        --model_args pretrained=google/gemma-1.1-7b-it,tensor_parallel_size=2,dtype='auto',gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks flores_eng_Latn-fuv_Latn_prompt_1,flores_eng_Latn-fuv_Latn_prompt_2,flores_eng_Latn-fuv_Latn_prompt_3,flores_fuv_Latn-eng_Latn_prompt_1,flores_fuv_Latn-eng_Latn_prompt_2,flores_fuv_Latn-eng_Latn_prompt_3 \
        --batch_size 'auto' \
        --device 'cuda' \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --limit 2

lm_eval --model vllm \
        --model_args pretrained=google/gemma-2-27b-it,tensor_parallel_size=2,dtype='auto',gpu_memory_utilization=0.9,data_parallel_size=1 \
        --tasks masakhapos_twi_prompt_3,masakhapos_wol_prompt_3,masakhapos_xho_prompt_3,masakhapos_yor_prompt_3,masakhapos_zul_prompt_3\
        --batch_size 'auto' \
        --num_fewshot 5 \
        --verbosity DEBUG \
        --output_path './masakhapos_results/' \
        --log_samples

lm_eval --model hf \
        --model_args pretrained=bigscience/mt0-small,parallelize=true \
        --tasks  injongointent_amh_prompt_1,injongointent_eng_prompt_1,injongointent_yor_prompt_1,injongointent_ibo_prompt_1,injongointent_wol_prompt_1\
        --device 'mps'  \
        --batch_size 1 \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --limit 5

lm_eval --model hf \
        --model_args pretrained=google/gemma-3-27b-it,parallelize=true \
        --tasks  afrobench_sentiment_tasks\
        --device 'cuda'  \
        --batch_size 1 \
        --num_fewshot 0 \
        --verbosity DEBUG \
        --output_path './senti_results/' \
        --log_samples
