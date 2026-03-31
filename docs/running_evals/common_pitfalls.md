# Common Pitfalls and Troubleshooting

This document highlights common pitfalls and troubleshooting tips when using the evaluation harness.

## YAML Configuration Issues

### Newline Characters in YAML (`\n`)

**Problem:** When specifying newline characters in YAML, they may be interpreted incorrectly depending on how you format them.

```yaml
# WRONG: Single quotes don't process escape sequences
generation_kwargs:
  until: ['\n']  # Gets parsed as the literal characters '\' and 'n' i.e "\\n"
```

```yaml
# RIGHT: Use double quotes for escape sequences
generation_kwargs:
  until: ["\n"]  # Gets parsed as an actual newline character
```

**Solutions:**

- Use double quotes for strings containing escape sequences
- For multiline content, use YAML's block scalars (`|` or `>`)
- When generating YAML programmatically, be careful with how template engines handle escape sequences

### Quoting in YAML

**When to use different types of quotes:**

- **No quotes**: Simple values (numbers, booleans, alphanumeric strings without special characters)

  ```yaml
  simple_value: plain text
  number: 42
  ```

- **Single quotes (')**:
  - Preserves literal values
  - Use when you need special characters to be treated literally
  - Escape single quotes by doubling them: `'It''s working'`

  ```yaml
  literal_string: 'The newline character \n is not processed here'
  path: 'C:\Users\name'  # Backslashes preserved
  ```

- **Double quotes (")**:
  - Processes escape sequences like `\n`, `\t`, etc.
  - Use for strings that need special characters interpreted
  - Escape double quotes with backslash: `"He said \"Hello\""`

  ```yaml
  processed_string: "First line\nSecond line"  # Creates actual newline
  unicode: "Copyright symbol: \u00A9"  # Unicode character
  ```

### Jinja2 in YAML

When using Jinja2 templates in `doc_to_text` or other fields, be careful with curly braces:

```yaml
# WRONG: Unquoted value with {{ — YAML tries to parse it as a mapping
doc_to_text: Question: {{question}}

# RIGHT: Quote the entire value
doc_to_text: "Question: {{question}}\nAnswer:"

# Also RIGHT: Use a block scalar
doc_to_text: |
  Question: {{question}}
  Answer:
```

## Evaluation Issues

### `--limit` is for testing only

The `--limit` flag restricts the number of examples evaluated per task. Results with `--limit` are **not comparable** to full evaluations and should never be reported as benchmark scores.

```bash
# Good: for testing your setup
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag --limit 10

# Bad: reporting these results as benchmark performance
```

### Unexpected metric values

If you see metrics that are very different from expected:

1. **Check the output type**: Is your task using `multiple_choice` when it should use `generate_until`, or vice versa?
2. **Check few-shot count**: `--num_fewshot 0` and omitting `--num_fewshot` may behave differently if the task YAML sets a default
3. **Check the prompt**: Use `--write_out` to inspect the actual prompts sent to the model
4. **Check the filter**: Some tasks apply post-processing filters (regex extraction, etc.) that may not match your model's output format

### Chat template issues

If instruction-tuned models perform worse than expected:

- Make sure you're using `--apply_chat_template`
- Check that the tokenizer includes the correct chat template
- See the [Chat Templates](chat_templates.md) guide for details

## Debugging

### Enable verbose logging

```bash
export LMEVAL_LOG_LEVEL="DEBUG"
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag --limit 5
```

### Inspect prompts

Use `--write_out` to print the actual prompts for the first few documents:

```bash
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag --write_out --limit 5
```

### Validate task configs

Before running a full evaluation, validate your task configurations:

```bash
lm-eval validate --tasks my_custom_task --include_path ./my_tasks/
```

## Where next?

- Debugging a task config? See [Task Configuration Reference](../writing_tasks/task_config_reference.md)
- Chat template issues? See [Chat Templates](chat_templates.md)
- Need help with scoring or filters? See [Scoring & Metrics](../writing_tasks/scoring_and_metrics.md)
