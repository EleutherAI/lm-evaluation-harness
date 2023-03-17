# `lm-evaluation-harness` + `promptsource`

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview

This project provides a unified framework to test causal (GPT-2, GPT-3, GPTNeo, etc) and seq2seq (T5, T0) language models via prompt evaluation.

As of now, all prompts are provided via the `promptsource` [eval-hackathon branch](https://github.com/bigscience-workshop/promptsource/tree/eval-hackathon); all datasets are from huggingface `datasets`.

This fork is __not__ backwards compatible with the original evaluation harness.

## Installation

```bash
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev]"
```

## CLI Usage 🖥️

To evaluate a model (e.g. GPT-2) on NLP tasks such as SuperGLUE WiC, you can run the following command:

```bash
python main.py \
    --model_api_name 'hf-causal' \
    --model_args pretrained='gpt2' \
    --task_name 'wic' \
    --template_names 'same_sense','polysemous' \
    --device cpu
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. For larger models supported by HuggingFace `transformers`, we provide parallelism and mixed-precision utilities through the [`accelerate`](https://github.com/huggingface/accelerate) package. It can be activated for `hf-causal`/`hf-seq2seq` by passing `use_accelerate=True` and `dtype=half` to the `--model_args` flag, respectively. For finer grained control over `accelerate` options, see the constructor docstrings for `HuggingFaceAutoLM` in `huggingface.py`.

```bash
python main.py \
    --model_api_name 'hf-causal' \
    --model_args use_accelerate=True,pretrained='facebook/opt-13b' \
    --task_name wnli
```

If you have access to the OpenAI API, you can also evaluate GPT-3 engines:

```bash
export OPENAI_API_SECRET_KEY={YOUR_KEY_HERE}
python main.py \
    --model_api_name 'openai' \
    --model_args engine='curie' \
    --task_name hans
```

 **When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility.** This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible.

### Detailed Usage

```
usage: main.py [-h] --model_api_name MODEL_API_NAME [--model_args MODEL_ARGS] --task_name TASK_NAME
               [--template_names TEMPLATE_NAMES] [--num_fewshot NUM_FEWSHOT] [--batch_size BATCH_SIZE]
               [--device DEVICE] [--limit LIMIT] [--output_path OUTPUT_PATH] [--template_idx TEMPLATE_IDX]
               [--bootstrap_iters BOOTSTRAP_ITERS] [--no_tracking] [--use_cache]

optional arguments:
  -h, --help            show this help message and exit
  --model_api_name MODEL_API_NAME
                        Name of the model API to use. See `lm_eval.list_model_apis()` for available APIs
  --model_args MODEL_ARGS
                        Model constructor args that you'd pass into a model of type `--model_api_name`. These must
                        be comma-separated keyword args, e.g. `key1=value1,key2=value2`, with no spaces
  --task_name TASK_NAME
                        Name of the task to use as found in the lm_eval registry. See: `lm_eval.list_tasks()`
  --task_args TASK_ARGS
                        Optional task constructor args that you'd pass into a task class of kind " `--task_name`.
                        These must be comma-separated keyword args, e.g. `key1=value1,key2=value2`, with no spaces.
                        WARNING: To avoid parsing errors, ensure your strings are quoted. For example,
                            `example_separator='\n+++\n'`
                        WARNING: Values must NOT contain commas.
  --template_names TEMPLATE_NAMES
                        Comma-separated list of template names for the specified task. Example:
                        `> python main.py ... --task_name rte --template_names imply,mean`
                        - Default: `all_templates`
                        - General Selectors:
                            - `"all_templates"`: Selects all templates for the task
                            - `"original_templates"`: Selects only templates that are designed to match the original task
  --num_fewshot NUM_FEWSHOT
  --batch_size BATCH_SIZE
  --seed SEED
  --device DEVICE       The device to place your model onto, e.g. cuda:0. For large models available through the
                        HuggingFace Hub you should use `accelerate` by passing `use_accelerate=True` to
                        `--model_args`
  --limit LIMIT         Limit the number of examples to evaluate on; ONLY USE THIS FOR DEBUGGING PURPOSES
  --output_path OUTPUT_PATH
                        Use output_path as `output_filename`. For example:
                        `> python main.py ... --output_path blop`
                        # saves files into `outputs/blop.json` Warning: You currently cannot change/add folder
                        structure.
  --template_idx TEMPLATE_IDX
                        Choose template by index from available templates
  --bootstrap_iters BOOTSTRAP_ITERS
                        Iters for stderr computation
  --no_tracking         Skip carbon emission tracking
  --use_cache           Whether to cache your model's predictions or not
```

## Library Usage 📖

You can also use `lm_eval` as a library:

```python
import lm_eval

model = lm_eval.get_model("hf-causal", pretrained="gpt2", device="cpu")
tasks = lm_eval.get_task_list(
    "superglue_rte",
    template_names=["does this imply", "must be true"])
results = lm_eval.evaluate(model=model, tasks=tasks)
```

The main user-facing functions are:

- [`lm_eval.get_model(model_api_name, **kwargs)`](./lm_eval/models/__init__.py) creates a model from a model API
- [`lm_eval.get_task(task_name, template_name, **kwargs)`](./lm_eval/tasks/__init__.py) creates a task with the prompt template
- [`lm_eval.get_task_list(task_name, template_names, **kwargs)`](./lm_eval/tasks/__init__.py) creates multiple instances of a task with different prompt templates
- [`lm_eval.evaluate(model, tasks, **kwargs)`](./lm_eval/evaluator.py) evaluates a model on a list of tasks

Some high-level convenience functions are also made available:
- [`lm_eval.list_model_apis()`](./lm_eval/models/__init__.py) lists all available model APIs you can evaluate from
- [`lm_eval.list_tasks()`](./lm_eval/tasks/__init__.py) lists all available tasks
- [`lm_eval.list_templates(task_name)`](./lm_eval/tasks/__init__.py) lists all available templates for a task
- [`lm_eval.get_templates(task_name)`](./lm_eval/tasks/__init__.py) returns promptsource dataset templates for a task

## Gotchas 🩹

- __You must pass templates to `PerplexityTask`s__  even though they will be ignored, as models will be scored from the raw text found in the task's dataset.

- __Multi-lingual ROUGE is unsupported__ as general token splitting is absent from [rouge-score](https://github.com/google-research/google-research/tree/master/rouge). For multi-lingual tasks, please ignore rouge metrics until this is resolved. _NOTE_: `English` works as intended.

- __Task versioning is not fully integrated__! If you're reporting your model's results, please include the package versions or commit IDs for this `lm-evaluation-harness` branch as well as the HuggingFace `datasets` and `promptsource` packages.

- __`promptsource` installation issue__: Some prompts may be excluded from the installed `promptsource` branch due to git-based pip installation issues. If the latest commit on the `promptsource/eval-hackathon` branch contains a prompt you're looking for but was not included in the installed version from our `setup.py`, you should run the following from within your environment:
    ```bash
    pip uninstall promptsource
    git clone --single-branch --branch eval-hackathon https://github.com/bigscience-workshop/promptsource
    cd promptsource
    pip install -e .
    ```

## Features

- Growing number of tasks integrated with `promptsource` (20+).

- Support for HuggingFace Causal language models, HuggingFace Seq2Seq models, and the OpenAI Completions API (GPT-3), with flexible tokenization-agnostic interfaces.

## Implementing new tasks

To implement a new task in eval harness, follow the [`PromptSourceTask` template](./templates/new_prompt_source_task.py).

## Using load_from_disk instead of load_dataset

You can use load_from_disk (convenient on Jean Zay supercomputer) by setting `task_args download_mode='load_from_disk',data_dir=<data/path>`
