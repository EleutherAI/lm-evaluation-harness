
model_name="google/gemma-1.1-2b-it"

accelerate launch -m --num_machines 1 --num_processes 2 --main_process_port 29507 lm_eval \
 --model hf \
--model_args pretrained="$model_name",dtype=bfloat16 \
--tasks noticia \
--device cuda:0 \
--batch_size auto \
--output_path ./outputs/"${model_name//\//_}" --write_out --log_samples
