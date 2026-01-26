TRITON_PRINT_AUTOTUNING=1 accelerate launch -m lm_eval \
    --model hf \
    --model_args dtype=bfloat16,pretrained=$1,max_length=32768 \
    --tasks longbench_narrativeqa,longbench_qasper,longbench_multifieldqa_en_e,longbench_hotpotqa,longbench_2wikimqa,longbench_musique,longbench_gov_report,longbench_qmsum,longbench_multi_news,longbench_trec,longbench_triviaqa,longbench_samsum,longbench_lcc,longbench_repobench-p \
    --batch_size 1
