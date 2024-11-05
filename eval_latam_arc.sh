PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
lm_eval --model hf \
--model_args pretrained=meta-llama/Llama-3.2-1B  \
--tasks latam_arc \
--device cuda:3 \
--batch_size 32 \
--num_fewshot 25 \
--output_path outputs \
--log_samples