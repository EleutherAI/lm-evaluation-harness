# Promptsource X Language Model Evaluation Harness

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview

This project provides a unified framework to test language models (GPT-2, GPT-3, GPTNeo, etc) and seq2seq (T5, T0) models via prompt evaluation.

As of now, all the prompts are provided via the `promptsource` [eval-hackathon branch](https://github.com/bigscience-workshop/promptsource/tree/eval-hackathon); all datasets are from huggingface `datasets`.

This fork is __not__ backwards compatible with the original evaluation harness.

## Installation

```bash
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
pip install   "promptsource @ git+https://github.com/bigscience-workshop/promptsource@eval-hackathon"
pip install -e ".[dev]"
```

## CLI Usage

To evaluate a model (e.g. GPT-2) on NLP tasks such as SuperGLUE, you can run the following command.

```bash
python main.py \
	--model hf-causal \
    --model_args pretrained=gpt2 \
	--tasks wic,copa
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. For larger models supported by HuggingFace `transformers`, we provide parallelism and mixed-precision utilities through the [`accelerate`](https://github.com/huggingface/accelerate) package. It can be activated for `hf-causal`/`hf-seq2seq` by passing `use_accelerate=True` and `dtype=half` to the `--model_args` flag, respectively. For finer grained control over `accelerate` options, see the constructor docstrings for `HuggingFaceAutoLM` in `huggingface.py`.

```bash
python main.py \
    --model hf-causal \
    --model_args use_accelerate=True,pretrained=facebook/opt-13b \
    --tasks wnli \
```

If you have access to the OpenAI API, you can also evaluate GPT-3 engines:


```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
	--model openai \
	--model_args engine=davinci \
	--tasks hans
```

 **When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility.** This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](https://github.com/EleutherAI/lm-evaluation-harness#task-versioning) section for more info.


Features:

- Growing number of tasks integrated with `promptsource` (20+).
- Support for hugging face causal language models, huggingface Seq2seq models, and the openai completion api (gpt3), with flexible tokenization-agnostic interface
- Task versioning to ensure reproducibility


## Implementing new tasks

To implement a new task in eval harness, follow the [`PromptSourceTask` template](./docs/task_guide.md).

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
