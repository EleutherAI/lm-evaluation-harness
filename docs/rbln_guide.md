# RBLN NPU Evaluation Guide

This guide covers the `rbln` and `rbln-vlm` model backends, which run language
and vision-language models on [Rebellions](https://rebellions.ai) NPUs (ATOM,
REBEL) via [`optimum-rbln`](https://github.com/rebellions-sw/optimum-rbln).

For a brief introduction and the basic CLI invocations, see the
[RBLN section in the main README](../README.md#rebellions-npu-inference-with-rbln--rbln-vlm).
This document is the model-type → recommended-task matrix and lists current
limitations.

## Installation

```bash
pip install "lm_eval[hf]" optimum[rbln]
```

`optimum-rbln` is an optional dependency — `lm_eval` still imports cleanly on
machines without the SDK installed; the backend simply becomes unavailable.

## Which flag to use?

The flag describes the **artifact type**, not the task:

- VLM-compiled artifact → **always `--model rbln-vlm`** (including text-only tasks)
- Causal / seq2seq LM artifact → `--model rbln`

## Supported model types

### 1. Causal LMs

**Examples:** LLaMA, GPT, Mistral, Qwen, Phi, Falcon, BLOOM, OPT
**Request types:** `loglikelihood`, `loglikelihood_rolling`, `generate_until`

The backend automatically detects when an RBLN-compiled artifact returns only
last-token logits and performs individual forward passes for each continuation
token, reconstructing full-sequence logits so `loglikelihood` results match
`--model hf` within numerical noise.

**Recommended tasks:**
- Multiple-choice / loglikelihood: `hellaswag`, `lambada_openai`, `arc_easy`, `arc_challenge`, `winogrande`, `truthfulqa_mc1`, `truthfulqa_mc2`, `mmlu`
- Generation: `gsm8k`

```bash
lm_eval --model rbln \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks hellaswag

lm_eval --model rbln \
    --model_args pretrained=gpt2 \
    --tasks lambada_openai
```

### 2. Seq2Seq LMs

**Examples:** T5, BART, Pegasus, Marian, mBART, BlenderBot
**Request types:** `loglikelihood`, `loglikelihood_rolling`, `generate_until`

The backend applies a `cache_position` scalar/tensor shim required by the RBLN
SDK and generates explicit `decoder_input_ids` / `decoder_attention_mask` with
float32 dtype.

**Recommended tasks:**
- Classification (GLUE): `cola`, `sst2`, `mrpc`, `qqp`, `mnli`, `qnli`, `rte`, `wnli`
- Generation: `squadv2`, `drop`, `gsm8k`

```bash
lm_eval --model rbln \
    --model_args pretrained=t5-small \
    --tasks cola

lm_eval --model rbln \
    --model_args pretrained=google/flan-t5-base \
    --tasks squadv2
```

### 3. Vision-Language Models (`rbln-vlm`)

**Examples:** LLaVA, LLaVA-NeXT, Qwen2-VL / Qwen2.5-VL / Qwen3-VL, Gemma3,
IDEFICS3, PaliGemma / PaliGemma2, Pixtral, BLIP-2

**Supported `model_type` values** (built-in compile profile):
`llava`, `llava_next`, `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `gemma3`,
`idefics3`, `paligemma`, `paligemma2`, `pixtral`, `blip-2` / `blip_2`.

**Request types:**
- Multimodal `generate_until` (e.g. `mmmu*`, `chartqa`)
- Text-only `loglikelihood` and `generate_until` on the **same** VLM
  artifact — no separate text-only checkpoint required

Text-only `loglikelihood` is implemented via
`model.generate(max_new_tokens=1, output_scores=True)` with
`token_type_ids=zeros` per continuation token, because direct
`model(input_ids=...)` is not a stable forward API on RBLN-compiled VLM
artifacts (state-init constraints). Scoring uses no `<bos>` prepend and no
chat template, so results stay directly comparable to a text-only LM
evaluated with `--model rbln`.

Each supported `model_type` ships with a built-in compile profile
(`_VLM_COMPILE_PROFILES` in `lm_eval/models/optimum_rbln.py`) mirroring the
[rbln-model-zoo image-text-to-text examples](https://github.com/rebellions-sw/rbln-model-zoo).
You can override any compile parameter via
`--model_args 'rbln_config_json={...}'` — the JSON is deep-merged into the
built-in profile.

```bash
# MMMU on Qwen2.5-VL
lm_eval --model rbln-vlm \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks mmmu_art --batch_size 1

# ChartQA on LLaVA 1.5
lm_eval --model rbln-vlm \
    --model_args pretrained=llava-hf/llava-1.5-7b-hf \
    --tasks chartqa --batch_size 1

# Text-only loglikelihood on a Gemma3 VLM artifact
lm_eval --model rbln-vlm \
    --model_args pretrained=/path/to/gemma-3-4b-it \
    --tasks hellaswag --batch_size 1

# Override compile profile (e.g. tensor_parallel_size=4)
lm_eval --model rbln-vlm \
    --model_args 'pretrained=Qwen/Qwen2.5-VL-7B-Instruct,rbln_config_json={"tensor_parallel_size":4}' \
    --tasks mmmu_art
```

## Multi-NPU (tensor parallel)

Use `rbln_tensor_parallel_size` for models that exceed a single NPU's memory:

```bash
lm_eval --model rbln \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,rbln_tensor_parallel_size=4 \
    --tasks hellaswag
```

## Current limitations

- **Masked LMs** (BERT, RoBERTa, DistilBERT) are not yet supported. The
  backend raises a clear error listing the supported `model_type` values when
  an unsupported architecture is requested.
- **Audio models** (Wav2Vec, Whisper) are not yet supported.
- **Multimodal `loglikelihood`** (requests carrying
  `aux_arguments["visual"]`) is not yet supported on `rbln-vlm`. Use
  generation-style multimodal tasks instead.

For the full list of RBLN-compatible models, see the
[Rebellions Model Zoo](https://rebellions.ai/developers/model-zoo/).
