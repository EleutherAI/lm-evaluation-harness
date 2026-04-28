# New Model Template

A copy-pasteable starting point for adding a new local-LM backend to `lm-evaluation-harness`. For the full walkthrough see [`docs/model_guide.md`](../../docs/model_guide.md).

## How to use

1. Copy `template.py` into `lm_eval/models/<your_model_filename>.py`.

   ```sh
   cp templates/new_template_model/template.py lm_eval/models/my_model.py
   ```

2. Pick the right base class:
   - **Subclassing `lm_eval.api.model.TemplateLM`** is usually what you want. It handles prompt assembly, sliding-window perplexity (`loglikelihood_rolling`), and basic batching for you — you only need to implement tokenization, `_loglikelihood_tokens`, and `generate_until`.
   - **Subclassing `lm_eval.api.model.LM`** directly is for backends that don't tokenize locally (e.g. remote APIs that take strings). You then have to implement all three request methods yourself.
   - **Subclassing `lm_eval.models.huggingface.HFLM`** is the easiest path if your model loads via `transformers.AutoModelForCausalLM` (or a close variant). See `lm_eval/models/mistral3.py` for a minimal example.

3. Update the `@register_model("...")` decorator with the CLI name(s) for your model.

4. Import your file from `lm_eval/models/__init__.py` so the registry sees it. Without this step, `--model my-model` will fail with "model not found".

5. Smoke-test on a tiny task before opening the PR:

   ```sh
   lm_eval --model my-model \
     --model_args pretrained=<path-or-id> \
     --tasks sciq \
     --limit 5 \
     --output_path results/my-model.json \
     --log_samples
   ```

   Verify the per-doc samples in `results/samples_*.jsonl` show real prompts and outputs.

## What the template demonstrates

- **Required interface**: `loglikelihood`, `loglikelihood_rolling`, `generate_until`, plus `eot_token_id`, `max_length`, `max_gen_toks`, `batch_size`.
- **`Collator` usage**: length-sorted batching with `group_by="gen_kwargs"` so requests sharing the same generation parameters stay together.
- **Partial caching**: how to call `self.cache_hook.add_partial(...)` so later runs benefit from `--use_cache`.
- **Optional chat-template hooks**: `tokenizer_name`, `chat_template`, `apply_chat_template`.

## Reference implementations

| Backend | File | Notes |
|---|---|---|
| HuggingFace `transformers` | `lm_eval/models/huggingface.py` | Full-featured reference — Collator, Reorderer, partial caching, chat templates, multi-GPU |
| Minimal HF subclass | `lm_eval/models/mistral3.py` | ~130 lines, shows how to override only what changes |
| Anthropic API | `lm_eval/models/anthropic_llms.py` | Remote-API style, no local tokenization |
| OpenAI-compatible | `lm_eval/models/openai_completions.py` | Local & remote completions endpoints |
