# Language Model Evaluation Harness

## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- Over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for commercial APIs including [OpenAI](https://openai.com), [goose.ai](https://goose.ai), and [TextSynth](https://textsynth.com/).
- Support for evaluation on adapters (e.g. LoRA) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Support for local models and benchmarks.
- Evaluation with publicly available prompts ensures reproducibility and comparability between papers.

The Language Model Evaluation Harness is the backend for 🤗 Hugging Face's popular [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) and is used internally by dozens of companies including NVIDIA, Cohere, Booz Allen Hamilton, and Mosaic ML.


## Install

To install the `lm-eval` package from the github repository, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages, you must install the package with the `multilingual` extra:

```bash
pip install -e ".[multilingual]"
```

To support loading GPTQ quantized models, install the package with the `gptq` extra:

```bash
pip install -e ".[gptq]"
```


To install the package with all extras, run
```bash
pip install -e ".[all]"
```

## Support

The best way to get support is to open an issue on this repo or join the EleutherAI discord server](discord.gg/eleutherai). The `#lm-thunderdome` channel is dedicated to developing this project and the `#release-discussion` channel is for receiving support for our releases.

## Basic Usage

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command:


```bash
python -m lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints, or to specify the datatype for running a model:

```bash
python -m lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Models that are loaded via both `transformers.AutoModelForCausalLM` (autoregressive, decoder-only GPT style models) and `transformers.AutoModelForSeq2SeqLM` (such as encoder-decoder models like T5) in Huggingface are supporteded.

Batch size selection can be automated by setting the  ```--batch_size``` flag to ```auto```. This will perform automatic detection of the largest batch size that will fit on your device. On tasks where there is a large difference between the longest and shortest example, it can be helpful to periodically recompute the largest batch size, to gain a further speedup. To do this, append ```:N``` to above flag to automatically recompute the largest batch size ```N``` times. For example, to recompute the batch size 4 times, the command would be:

```bash
python -m lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size auto:4
```

Alternatively, you can use `lm-eval` or `lm_eval` instead of `python -m lm_eval` to call lm eval from anywhere.

### Multi-GPU Evaluation with Hugging Face `accelerate`

To parallelize evaluation of HuggingFace models across multiple GPUs, we allow for two different types of multi-GPU evaluation.

The first is performed by launching evaluation via the `accelerate` library as follows:

```
accelerate launch -m lm_eval \
    --model hf \
    --tasks lambada_openai,arc_easy \
    --batch_size 16 \
```

This will perform *data-parallel evaluation*: that is, placing a **single full copy** of your model onto each available GPU and *splitting batches across GPUs* to evaluate on K GPUs K times faster than on one.

If your model is *is too large to be run on a single one of your GPUs* then you can use `accelerate` with Fully Sharded Data Parallel (FSDP) that splits the weights of the model across your data parallel ranks. To enable this, ensure you select `YES` when asked ```Do you want to use FullyShardedDataParallel?``` when running `accelerate config`. To enable memory-efficient loading, select `YES` when asked `Do you want each individually wrapped FSDP unit to broadcast module parameters from rank 0 at the start?`. This will ensure only the rank 0 process loads the model and then broadcasts the parameters to the other ranks instead of having each rank load all parameters which can lead to large RAM usage spikes around the start of the script that may cause errors.


We also provide an second method to run these large models: use of the `parallelize` argument.
```
python -m lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-12b,parallelize=True
    --tasks lambada_openai,arc_easy \
    --batch_size 16
```

To pass even more advanced keyword arguments to `accelerate`, we allow for the following arguments as well:
- `device_map_option`: How to split model weights across available GPUs. defaults to "auto".
- `max_memory_per_gpu`: the max GPU memory to use per GPU in loading the model.
- `max_cpu_memory`: the max amount of CPU memory to use when offloading the model weights to RAM.
- `offload_folder`: a folder where model weights will be offloaded to disk if needed.

Note that this method naively splits models across GPUs, resulting in only a single GPU performing work at any point in time, and so is much slower than launching with `accelerate launch`, possibly by a factor of the total # of GPUs.

**Note that this option requires launching evaluation via `python -m lm_eval` rather than `accelerate launch -m lm_eval`.**

To use `accelerate` with the `lm-eval` command, use
```
accelerate launch --no_python lm-eval --model ...
```

### Commercial APIs

Our library also supports the evaluation of models served via several commercial APIs, and hope to implement support for common performant local/self-hosted inference servers.

A full accounting of the supported and planned libraries + APIs can be seen below:

| API or Inference Server     | Implemented?                    | `--model <xxx>` name                                                             | Models supported:                    | Request Types:                                           |
|-----------------------------|---------------------------------|----------------------------------------------------------------------------------|--------------------------------------|----------------------------------------------------------|
| OpenAI Completions          | :heavy_check_mark:              | `openai`, `openai-completions`, `gooseai`                                        | up to `code-davinci-002`             | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| OpenAI ChatCompletions      | :x: Not yet - needs testing!       | N/A                                                                              | [All ChatCompletions API models](https://platform.openai.com/docs/guides/gpt)                         | `generate_until` (no logprobs)                             |
| Anthropic                   | :heavy_check_mark:              | `anthropic`                                                                      | [Supported Anthropic Engines](https://docs.anthropic.com/claude/reference/selecting-a-model)         | `generate_until` (no logprobs)                             |
| GooseAI                     | :heavy_check_mark: (not separately maintained)  | `openai`, `openai-completions`, `gooseai` (same interface as OpenAI Completions) |                                      | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Textsynth                   | Needs testing                   | `textsynth`                                                                      | ???                                  | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| Cohere                      | :hourglass: - blocked on Cohere API bug | N/A                                                                              | [All `cohere.generate()` engines](https://docs.cohere.com/docs/models) | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| GGML                        | :hourglass: [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/617)              | N/A                                                                              | ???                                  | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| vLLM                        | :x: Not yet - needs help!       | N/A                                                                              | All HF models                        | `generate_until` (no logprobs)                             |
| Your inference server here! | ...                             | ...                                                                              | ...                                  | ...                                                      |                                | ...                                                      |

It is on our roadmap to create task variants designed to enable models which do not serve logprobs/loglikelihoods to be compared with generation performance of open-source models.

Our library supports language models served via the OpenAI Completions API as follows:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python -m lm_eval \
    --model openai-completions \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially maintained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library. These include [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md), and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

### Additional Features

If you have a CUDA-compatible Mac GPU, you can run the eval harness using the MPS back-end by replaicng `--device cuda:0` with `--device mps:0`. PyTorch does not currently support automatic mixed precision (AMP) for MPS, so we forcibly cast all weights to fp32 regardless of how they're stored. This is slower and has a larger memory footprint than we can achieve on Linux systems, but as PyTorch continues to improve its MPS support we hope to continue to improve it.

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python -m lm_eval \
    --model openai \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

## Advanced Usage

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
python -m lm_eval \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

[GPTQ](https://github.com/PanQiWei/AutoGPTQ) quantized models can be loaded by specifying their file names in `,gptq=NAME` (or `,gptq=True` for default names) in the `model_args` argument:

```bash
python -m lm_eval \
    --model hf \
    --model_args pretrained=model-name-or-path,gptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.


## How to Contribute or Learn More?

For more information on the library and how everything fits together, check out all of our [documentation pages](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor/docs)! We plan to post a larger roadmap of desired + planned library improvements soon, with more information on how contributors can help.


You can also ask for help, or discuss new features with the maintainers in the #lm-thunderdome channel of the EleutherAI discord! If you've used the library and have had a positive (or negative) experience, we'd love to hear from you!

### Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/new_task_guide.md).


As a start, we currently only support one prompt per task, which we strive to make the "standard" as defined by the benchmark's authors. If you would like to study how varying prompts causes changes in the evaluation score, we support prompts authored in the [Promptsource Library](https://github.com/bigscience-workshop/promptsource/tree/main) as described further in [the task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/docs/new_task_guide.md) and [the advanced task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/docs/advanced_task_guide.md) and welcome contributions of novel task templates and task variants.


## Cite as

```
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
