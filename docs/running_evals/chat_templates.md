# Chat Templates

This guide covers how chat templates interact with the evaluation harness when evaluating instruction-tuned and chat models.

## Overview

When evaluating chat/instruct models, prompts need to be formatted with the model's chat template (special tokens like `<|user|>`, `<|assistant|>`, etc.). The `--apply_chat_template` flag enables this.

```bash
lm-eval run --model hf --model_args pretrained=meta-llama/Llama-3-8B-Instruct \
    --tasks hellaswag \
    --apply_chat_template
```

## Delimiter handling

When `apply_chat_template=True`, the target delimiter is set to an empty string instead of the default whitespace. This prevents interference between chat template formatting and the delimiter system.

```text
# Without chat template (default delimiter " ")
Question: What color is the sky?
Answer: blue

# With chat template (empty delimiter)
<|user|>Question: What color is the sky?
Answer:<|assistant|>blue
```

This is important for multiple-choice tasks where the template itself handles spacing between the prompt and the answer choices.

## Using with few-shot examples

### Multi-turn formatting

When `--apply_chat_template` is enabled, few-shot examples are automatically formatted as multi-turn conversations (alternating user/assistant messages):

```bash
lm-eval run --model hf --model_args pretrained=meta-llama/Llama-3-8B-Instruct \
    --tasks arc_easy \
    --num_fewshot 5 \
    --apply_chat_template
```

This produces prompts like:

```text
<|user|>Question: What is H2O?
A. Hydrogen
B. Water
C. Oxygen
D. Salt<|assistant|>B<|user|>Question: ...
```

To disable multi-turn formatting while still using chat templates:

```bash
--apply_chat_template --fewshot_as_multiturn false
```

### System instructions

Add a system prompt that will be inserted at the beginning of the conversation:

```bash
lm-eval run --model hf --model_args pretrained=meta-llama/Llama-3-8B-Instruct \
    --tasks mmlu \
    --apply_chat_template \
    --system_instruction "You are a helpful assistant. Answer each question by selecting the correct option."
```

## Using with prompt formats

Chat templates and [prompt formats](../writing_tasks/prompt_formats.md) work together. You can apply a format to structure the question/answer layout, and the chat template to add the model's special tokens:

```bash
# Format structures the prompt content, chat template adds special tokens
lm-eval run --tasks my_task@mcqa \
    --model hf --model_args pretrained=meta-llama/Llama-3-8B-Instruct \
    --apply_chat_template
```

The format controls the textual layout (question, choices, answer solicitation), while the chat template wraps it in the model's conversation structure.

## Generation prefix

Use `gen_prefix` in your task YAML to append text after the `<|assistant|>` token:

```yaml
gen_prefix: "The answer is: "
```

This is useful for prompting the model to start its response in a specific way. Without a chat template, `gen_prefix` is appended to the end of the prompt instead.

## Completions vs. chat-completion endpoints

!!! note
    Loglikelihood and multiple-choice tasks (such as MMLU) are only supported for **completion** endpoints, not for chat-completion endpoints that expect a list of dicts. Completion APIs supporting instruct-tuned models can use `--apply_chat_template` to evaluate with a chat template format while still accessing the model logits needed for loglikelihood-based tasks.
