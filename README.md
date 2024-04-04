# How to test gpt-3.5 on wmdp

Run

```
pip install -e .
pip install openai tiktoken
lm_eval --model openai-chat-completions --model_args model=gpt-3.5-turbo-0125 --tasks wmdp --output_path out --log_samples --gen_kwargs until='.',max_gen_toks=10

```
