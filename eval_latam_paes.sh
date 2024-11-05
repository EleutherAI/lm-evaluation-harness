CUDA_VISIBLE_DEVICES=1,2,3,4
lm_eval --model hf \
--model_args "pretrained=meta-llama/Llama-3.1-70B,parallelize=True" \
--tasks paes \
--device cuda \
--batch_size 1 \
--num_fewshot 0 \
--output_path outputs \
--log_samples