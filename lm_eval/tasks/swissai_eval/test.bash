# Example command to test tasks in local

lm_eval --model hf --model_args pretrained=sshleifer/tiny-gpt2,device=cpu --limit 2 --log_samples --output_path out --tasks TASK_NAME