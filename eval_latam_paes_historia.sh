lm_eval --model hf \
--model_args "pretrained=meta-llama/Llama-3.1-70B-Instruct,parallelize=True" \
--tasks paes_historia \
--batch_size 16 \
--num_fewshot 0 \
--output_path outputs \
--log_samples
