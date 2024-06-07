lm_eval --model hf \
--model_args pretrained=/workspace1/sebcif/es-checkpoints/tmp-checkpoint-42000/ \
--tasks latam_hellaswag \
--device cuda:0 \
--batch_size 32 \
--num_fewshot 10 \
--output_path outputs \
--log_samples