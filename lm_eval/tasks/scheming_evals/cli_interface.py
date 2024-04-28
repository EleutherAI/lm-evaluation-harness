'''
### TESTING INTERFACE (without API use) ###
lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-2.8b \
    --include_path ./ \
    --tasks [TASK NAME] \
    --device mps \
    --limit 10 \
    --output output/[TASK OUTPUT FILE]/ \
    --log_samples \
    --predict_only 

### FREE RESPONSE TASK ###
export OPENAI_API_KEY=KEY
lm_eval \
    --model openai-chat-completions \
    --model_args model=gpt-4\
    --include_path path/to/scheming_evals \
    --tasks scheming_evals_free_response_task \
    --output output/scheming_evals_free_response_task_output/ \
    --log_samples \
    --predict_only 
    
    
### MC TASK ###
export OPENAI_API_KEY=KEY
lm_eval \
    --model openai-chat-completions \
    --model_args model=gpt-4\
    --include_path path/to/scheming_evals \
    --tasks scheming_evals_mc_prompt_task \
    --output output/scheming_evals_mc_task_output/ \
    --log_samples \
    --predict_only 
    
    
### GOAL TASK ###  
export OPENAI_API_KEY=KEY
lm_eval \
    --model openai-chat-completions \
    --model_args model=gpt-4\
    --include_path path/to/scheming_evals \
    --tasks scheming_evals_goal_task \
    --output output/scheming_evals_goal_task_output/ \
    --predict_only \
    --log_samples
    
### CONTEXT TASK ### 
export OPENAI_API_KEY=KEY
lm_eval \
    --model openai-chat-completions \
    --model_args model=gpt-4\
    --include_path path/to/scheming_evals \
    --tasks scheming_evals_context_task \
    --output output/scheming_evals_context_task_output/ \
    --log_samples \
    --predict_only
'''