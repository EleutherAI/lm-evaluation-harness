lm_eval --model hf \
--model_args "pretrained=meta-llama/Llama-3.1-70B-Instruct,parallelize=True"  \
--tasks latam_arc \
--batch_size 32 \
--num_fewshot 25 \
--output_path outputs \
--log_samples