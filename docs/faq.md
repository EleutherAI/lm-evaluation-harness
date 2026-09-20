# Frequently Asked Questions

This page collects short answers to common setup and evaluation questions. For deeper troubleshooting, see the [Common Pitfalls and Troubleshooting Guide](./footguns.md).

## How are evaluation request types implemented?

The harness routes task examples into model request types such as `loglikelihood`, `loglikelihood_rolling`, and `generate_until`. These are implemented by model wrappers that subclass `lm_eval.api.model.LM`.

See the [Model Guide](./model_guide.md#interface) for the method signatures and examples. For a longer treatment of evaluation details, see the paper appendix linked from issue [#1676](https://github.com/EleutherAI/lm-evaluation-harness/issues/1676).

## Why do offline runs try to access Hugging Face?

Some tasks and models resolve datasets, tokenizers, configs, or weights through Hugging Face unless you point the harness at local files or a local cache. If you need offline execution, prepare the relevant dataset/model artifacts ahead of time and pass local paths where supported by the task or model.

For task authoring, see the [New Task Guide](./new_task_guide.md). For model wrappers and tokenizer handling, see the [Model Guide](./model_guide.md).

## How do I use a dataset saved locally?

Use the task configuration fields that identify the dataset path and loading arguments for your task. Local paths are most often handled in YAML task configs, then selected with the normal task name or include path mechanisms.

Start with the [Task Configuration Guide](./task_guide.md) and the [New Task Guide](./new_task_guide.md), then compare against existing task YAMLs under `lm_eval/tasks/`.

## Why do stop sequences or newlines behave unexpectedly in YAML?

YAML quoting changes how escape sequences are interpreted. For example, single quotes preserve the literal characters `\` and `n`, while double quotes can encode an actual newline.

See [Newline Characters in YAML](./footguns.md#newline-characters-in-yaml-n) for concrete examples.

## Which guide should I read first?

- Use the [Interface Guide](./interface.md) to run evaluations from the command line or Python.
- Use the [New Task Guide](./new_task_guide.md) when adding a task.
- Use the [Task Configuration Guide](./task_guide.md) when editing YAML task behavior.
- Use the [Model Guide](./model_guide.md) when adding a model backend.
- Use the [API Guide](./API_guide.md) for API-served model integrations.
