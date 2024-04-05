'''
export OPENAI_API_KEY=YOUR_KEY_HERE
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-2.8b \
    --include_path /Users/marina.levay/Documents/GitHub/lm-evaluation-harness/lm_eval/tasks/scheming_evals \
    --tasks scheming_evals_free_response_task \
    --limit 10 \
    --output output/scheming_evals_free_response_task_output/ \
    --device mps \
    --log_samples \
    --predict_only \
    --verbosity DEBUG
    
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-2.8b \
    --include_path /Users/marina.levay/Documents/GitHub/lm-evaluation-harness/lm_eval/tasks/scheming_evals \
    --tasks scheming_evals_free_response \
    --limit 10 \
    --output output/scheming_evals_free_response_output/ \
    --device mps \
    --predict_only \
    --log_samples
    --verbosity DEBUG

lm_eval \
    --model openai-chat-completions \
    --model_args model=gpt-3.5-turbo \
    --include_path /Users/marina.levay/Documents/GitHub/lm-evaluation-harness/lm_eval/tasks/scheming_evals \
    --tasks scheming_evals_mc_prompt_task \
    --output output/scheming_evals_mc_prompt_task_output/ \
    --log_samples
    --verbosity DEBUG
    
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-2.8b \
    --include_path /Users/marina.levay/Documents/GitHub/lm-evaluation-harness/lm_eval/tasks/scheming_evals \
    --tasks scheming_evals_mc_prompt_task \
    --limit 10 \
    --output output/scheming_evals_mc_prompt_task_output/ \
    --device mps \
    --predict_only \
    --log_samples
    --verbosity DEBUG
    
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-2.8b \
    --include_path /Users/marina.levay/Documents/GitHub/lm-evaluation-harness/lm_eval/tasks/scheming_evals \
    --tasks scheming_evals_mc_prompt_task \
    --limit 10 \
    --output output/scheming_evals_mc_prompt_task_output/ \
    --device mps \
    --log_samples
    --verbosity DEBUG
    
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-2.8b \
    --include_path /Users/marina.levay/Documents/GitHub/lm-evaluation-harness/lm_eval/tasks/scheming_evals \
    --tasks scheming_evals_context_task \
    --limit 10 \
    --output output/scheming_evals_context_task_output/ \
    --device mps \
    --log_samples \
    --verbosity DEBUG
'''