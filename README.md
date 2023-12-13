# Language Model Evaluation Harness

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836)

## Announcement
**A new v0.4.0 release of lm-evaluation-harness is available** !

New updates and features include:

- Internal refactoring
- Config-based task creation and configuration
- Easier import and sharing of externally-defined task config YAMLs
- Support for Jinja2 prompt design, easy modification of prompts + prompt imports from Promptsource
- More advanced configuration options, including output post-processing, answer extraction, and multiple LM generations per document, configurable fewshot settings, and more
- Speedups and new modeling libraries supported, including: faster data-parallel HF model usage, vLLM support, MPS support with HuggingFace, and more
- Logging and usability changes
- New tasks including CoT BIG-Bench-Hard, Belebele, user-defined task groupings, and more

Please see our updated documentation pages in `docs/` for more details.

Development will be continuing on the `main` branch, and we encourage you to give us feedback on what features are desired and how to improve the library further, or ask questions, either in issues or PRs on GitHub, or in the [EleutherAI discord](discord.gg/eleutherai)!

## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

**Features:**
- Over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for fast and memory-efficient inference with [vLLM](https://github.com/vllm-project/vllm).
- Support for commercial APIs including [OpenAI](https://openai.com), [goose.ai](https://goose.ai), and [TextSynth](https://textsynth.com/).
- Support for evaluation on adapters (e.g. LoRA) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Support for local models and benchmarks.
- Evaluation with publicly available prompts ensures reproducibility and comparability between papers.
- Easy support for custom prompts and evaluation metrics.

The Language Model Evaluation Harness is the backend for 🤗 Hugging Face's popular [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), has been used in [hundreds of papers](https://scholar.google.com/scholar?oi=bibs&hl=en&authuser=2&cites=15052937328817631261,4097184744846514103,17476825572045927382,18443729326628441434,12854182577605049984) is used internally by dozens of companies including NVIDIA, Cohere, Nous Research, Booz Allen Hamilton, and Mosaic ML.

## Install

To install the `lm-eval` package from the github repository, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

We also provide a number of optional dependencies for extended functionality. Extras can be installed via `pip install -e ".[NAME]"`

| Name          | Use                                   |
| ------------- | ------------------------------------- |
| anthropic     | For using Anthropic's models          |
| dev           | You probably don't want to use this   |
| gptq          | For loading models with GPTQ          |
| testing       | You probably don't want to use this   |
| multilingual  | For multilingual tokenizers           |
| openai        | For using OpenAI's models             |
| promptsource  | For using PromtSource prompts         |
| sentencepiece | For using the sentencepiece tokenizer |
| vllm          | For loading models with vLLM          |
| all           | Loads all extras                      |

## Basic Usage

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints, or to specify the datatype for running a model:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Models that are loaded via both `transformers.AutoModelForCausalLM` (autoregressive, decoder-only GPT style models) and `transformers.AutoModelForSeq2SeqLM` (such as encoder-decoder models like T5) in Huggingface are supporteded.

Batch size selection can be automated by setting the  ```--batch_size``` flag to ```auto```. This will perform automatic detection of the largest batch size that will fit on your device. On tasks where there is a large difference between the longest and shortest example, it can be helpful to periodically recompute the largest batch size, to gain a further speedup. To do this, append ```:N``` to above flag to automatically recompute the largest batch size ```N``` times. For example, to recompute the batch size 4 times, the command would be:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size auto:4
```

Alternatively, you can use `lm-eval` instead of `lm_eval`.

> [!Note]
> Just like you can provide a local path to `transformers.AutoModel`, you can also provide a local path to `lm_eval` via `--model_args pretrained=/path/to/model`

#### Multi-GPU Evaluation with Hugging Face `accelerate`

To parallelize evaluation of HuggingFace models across multiple GPUs, we leverage the [accelerate 🚀](https://github.com/huggingface/accelerate) library as follows:

```
accelerate launch -m lm_eval --model hf \
    --tasks lambada_openai,arc_easy \
    --batch_size 16
```

This will perform *data-parallel evaluation*: that is, placing a **single full copy** of your model onto each available GPU and *splitting batches across GPUs* to evaluate on K GPUs K times faster than on one.

If your model is *is too large to be run on a single one of your GPUs* then you can use `accelerate` with Fully Sharded Data Parallel (FSDP) that splits the weights of the model across your data parallel ranks. To enable this, ensure you select `YES` when asked ```Do you want to use FullyShardedDataParallel?``` when running `accelerate config`. To enable memory-efficient loading, select `YES` when asked `Do you want each individually wrapped FSDP unit to broadcast module parameters from rank 0 at the start?`. This will ensure only the rank 0 process loads the model and then broadcasts the parameters to the other ranks instead of having each rank load all parameters which can lead to large RAM usage spikes around the start of the script that may cause errors.

To pass even more advanced keyword arguments to `accelerate`, we allow for the following arguments as well:
- `device_map_option`: How to split model weights across available GPUs. defaults to "auto".
- `max_memory_per_gpu`: the max GPU memory to use per GPU in loading the model.
- `max_cpu_memory`: the max amount of CPU memory to use when offloading the model weights to RAM.
- `offload_folder`: a folder where model weights will be offloaded to disk if needed.

To use `accelerate` with the `lm-eval` command, use
```
accelerate launch --no_python lm-eval --model ...
```


### Tensor + Data Parallel and Optimized Inference with `vLLM`

We also support vLLM for faster inference on [supported model types](https://docs.vllm.ai/en/latest/models/supported_models.html). For single-GPU or multi-GPU — tensor parallel, data parallel, or a combination of both — inference, for example:

```bash
lm_eval --model vllm \
    --model_args pretrained={model_name},tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas} \
    --tasks lambada_openai \
    --batch_size auto
```
For a full list of supported vLLM configurations, please reference our vLLM integration and the vLLM documentation.

vLLM occasionally differs in output from Huggingface. We treat Huggingface as the reference implementation, and provide a script at [./scripts/model_comparator.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/scripts/model_comparator.py) for checking validity of vllm results against HF.

### Model APIs and Inference Servers

Our library also supports the evaluation of models served via several commercial APIs, and we hope to implement support for the most commonly used performant local/self-hosted inference servers.

To call a hosted model, use:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
lm_eval --model openai-completions \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

Note that for externally hosted models, configs such as `--device` and `--batch_size` should not be used and do not function. Just like you can use `--model_args` to pass arbitrary arguments to the model constructor for local models, you can use it to pass arbitrary arguments to the model API for hosted models. See the documentation of the hosting service for information on what arguments they support.


| API or Inference Server     | Implemented?                    | `--model <xxx>` name                                                             | Models supported:                                                                             | Request Types:                                           |
|-----------------------------|---------------------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------|
| OpenAI Completions          | :heavy_check_mark:              | `openai`, `openai-completions`, `gooseai`                                        | up to `code-davinci-002`                                                                      | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| OpenAI ChatCompletions      | :x: Not yet - needs testing!       | N/A                                                                              | [All ChatCompletions API models](https://platform.openai.com/docs/guides/gpt)                 | `generate_until` (no logprobs)                             |
| Anthropic                   | :heavy_check_mark:              | `anthropic`                                                                      | [Supported Anthropic Engines](https://docs.anthropic.com/claude/reference/selecting-a-model)  | `generate_until` (no logprobs)                             |
| GooseAI                     | :heavy_check_mark: (not separately maintained)  | `openai`, `openai-completions`, `gooseai` (same interface as OpenAI Completions) |                                                                                               | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Textsynth                   | :heavy_check_mark:                   | `textsynth`                                                                      | [All supported engines](https://textsynth.com/documentation.html#engines)                                                                                           | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Cohere                      | [:hourglass: - blocked on Cohere API bug](https://github.com/EleutherAI/lm-evaluation-harness/pull/395) | N/A                                                                              | [All `cohere.generate()` engines](https://docs.cohere.com/docs/models)                        | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| [Llama.cpp](https://github.com/ggerganov/llama.cpp) (via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python))                        | :heavy_check_mark:              | `gguf`, `ggml`                                                                   | [All models supported by llama.cpp](https://github.com/ggerganov/llama.cpp)               | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| vLLM                        | :heavy_check_mark:       | `vllm`                                                                           | [Most HF Causal Language Models](https://docs.vllm.ai/en/latest/models/supported_models.html) | `generate_until`, `loglikelihood`, `loglikelihood_rolling`                             |
| Your inference server here! | ...                             | ...                                                                              | ...                                                                                           | ...                                                      |                                | ...                                                      |

It is on our roadmap to create task variants designed to enable models which do not serve logprobs/loglikelihoods to be compared with generation performance of open-source models.

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library. These include [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md), and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

### Additional Features

If you have a Metal compatible Mac, you can run the eval harness using the MPS back-end by replacing `--device cuda:0` with `--device mps` (requires PyTorch version 2.1 or higher).

> [!Note]
> You can inspect what the LM inputs look like by running the following command:
> ```bash
> python write_out.py \
>     --tasks all_tasks \
>     --num_fewshot 5 \
>     --num_examples 10 \
>     --output_base_path /path/to/output/folder
> ```
> This will write out one text file for each task.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
lm_eval --model openai \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

## Advanced Usage Tips

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

[GPTQ](https://github.com/PanQiWei/AutoGPTQ) quantized models can be loaded by specifying their file names in `,gptq=NAME` (or `,gptq=True` for default names) in the `model_args` argument:

```bash
lm_eval --model hf \
    --model_args pretrained=model-name-or-path,gptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

To save evaluation results provide an `--output_path`. We also support logging model responses with the `--log_samples` flag for post-hoc analysis.

Additionally, one can provide a directory with `--use_cache` to cache the results of prior runs. This allows you to avoid repeated execution of the same (model, task) pairs for re-scoring.

For a full list of supported arguments, check out the [interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) guide in our documentation!

## How to Contribute or Learn More?

For more information on the library and how everything fits together, check out all of our [documentation pages](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor/docs)! We plan to post a larger roadmap of desired + planned library improvements soon, with more information on how contributors can help.

### Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/new_task_guide.md).

In general, we follow this priority list for addressing concerns about prompting and other eval details:
1. If there is widespread agreement among people who train LLMs, use the agreed upon procedure.
2. If there is a clear and unambiguous official implementation, use that procedure.
3. If there is widespread agreement among people who evaluate LLMs, use the agreed upon procedure.
4. If there are multiple common implementations but not universal or widespread agreement, use our preferred option among the common implementations. As before, prioritize choosing from among the implementations found in LLM training papers.

These are guidelines and not rules, and can be overruled in special circumstances.

We try to prioritize agreement with the procedures used by other groups to decrease the harm when people inevitably compare runs across different papers despite our discouragement of the practice. Historically, we also prioritized the implementation from [Language Models are Few Shot Learners](https://arxiv.org/abs/2005.14165) as our original goal was specifically to compare results with that paper.

### Support

The best way to get support is to open an issue on this repo or join the [EleutherAI Discord server](https://discord.gg/eleutherai). The `#lm-thunderdome` channel is dedicated to developing this project and the `#release-discussion` channel is for receiving support for our releases. If you've used the library and have had a positive (or negative) experience, we'd love to hear from you!

## Cite as

```
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
```
