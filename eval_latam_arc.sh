lm_eval --model hf \
--model_args pretrained=/workspace1/sebcif/es-checkpoints/tmp-checkpoint-42000/ \
--tasks latam_arc \
--device cuda:0 \
--batch_size 32 \
--num_fewshot 25 \
--output_path outputs \
--log_samples