# Running K2 Model Evaluations with lm-eval

This guide covers:

1. Getting started with this repository
2. Running `MBZUAI-IFM/K2-V2-Instruct` from Hugging Face
3. Running `MBZUAI-IFM/K2-V2-Instruct` via the K2 API

## 1) Get Started with the Repository

Clone and install:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Install backend extras depending on how you want to run:

- Hugging Face backend:

```bash
pip install "lm_eval[hf]"
```

- API backend:

```bash
pip install "lm_eval[api]"
```

- Or install both:

```bash
pip install "lm_eval[hf,api]"
```

List available tasks:

```bash
lm-eval ls tasks
```

## 2) Run K2-V2-Instruct from Hugging Face

Example (GPU):

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=MBZUAI-IFM/K2-V2-Instruct \
  --tasks hellaswag \
  --device cuda:0,1 \
  --batch_size 8
```

## 3) Run K2-V2-Instruct from the API

Your endpoint is OpenAI Chat Completions-compatible, so use `local-chat-completions`.

```bash
export OPENAI_API_KEY="<your-api-key>"
```
```bash
lm-eval run \
  --model local-chat-completions \
  --model_args model=MBZUAI-IFM/K2-V2-Instruct,base_url=https://api.k2think.ai/v1/chat/completions,num_concurrent=4,max_retries=5 \
  --tasks gsm8k \
  --apply_chat_template
```

Important notes:

- `local-chat-completions` is best for generation-style tasks.
- In this harness, chat-completions backends do not support `loglikelihood`-style evaluation paths; prefer generation tasks (for example `gsm8k`) when using the API route.