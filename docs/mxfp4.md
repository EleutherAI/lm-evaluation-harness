# MXFP4 fake-quantization evaluation

The `hf-mxfp4` backend runs standard Harness tasks on floating-point Hugging Face
checkpoints with MXFP4 weight and activation quantize/dequantize (QDQ). It uses
PyTorch and the existing HF backend, with no veRL, ModelOpt, or native FP4 kernel
dependency. All three request types are available: `loglikelihood`,
`loglikelihood_rolling`, and `generate_until`.

This is an **accuracy reference**, not a packed FP4 inference engine. Weights and
matrix multiplications remain in the selected floating-point dtype. Dynamic
activation QDQ adds work; this backend does not promise faster inference or
4-bit model memory usage. W4A4 here means **MXFP4**, not INT4 or NVIDIA NVFP4.

## Installation and use

Install the Harness with its existing HF dependencies:

```bash
pip install -e ".[hf]"
```

Run a small public model on a standard task:

```bash
lm_eval --model hf-mxfp4 \
    --model_args pretrained=EleutherAI/pythia-14m,precision=w4a4,dtype=bfloat16 \
    --tasks boolq --device cpu --batch_size 1 --limit 8 \
    --output_path results/mxfp4-smoke
```

For a full accuracy comparison, use the same checkpoint revision, tasks, seeds,
few-shot count, batch size, dtype, and chat-template settings in separate runs:

```bash
for precision in bf16 w4a16 w4a4; do
    lm_eval --model hf-mxfp4 \
        --model_args pretrained=/path/to/model,precision=$precision,dtype=bfloat16 \
        --tasks arc_easy,hellaswag,winogrande \
        --device cuda:0 --batch_size 1 \
        --output_path results/$precision
done
```

Use a separate response cache path for each precision/configuration if enabling
`--use_cache`. A run with `--limit` is a smoke test, not an accuracy benchmark.

| Argument | Default | Meaning |
| --- | --- | --- |
| `precision` | `w4a4` | `bf16`: floating baseline; `w4a16`: weight QDQ only; `w4a4`: weight and Linear-input QDQ |
| `dtype` | `bfloat16` | Storage and floating compute dtype: `bfloat16`, `float16`, or `float32`; `bf16` mode requires `bfloat16` |
| `ignore` | `lm_head;*.gate;*.router` | Semicolon-separated glob patterns matched against full module names |

`w4a16` denotes the weight-only diagnostic: its activations use `dtype` unchanged
(including FP32 when explicitly selected for reference testing). `dtype=auto` and
additional autocast via `mixed_precision_dtype` are rejected so the computation
dtype is explicit. The normal HF options for tokenization, batching, generation,
and local checkpoints remain available.

## Numerical contract

The format is defined by [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
sections 5.3.3, 5.4.1 and 6.3. This backend selects the following concrete
reference recipe; other MXFP4 recipes can use different scaling/calibration.

- Contiguous groups of **32** along the last dimension, independently for each
  weight row or input token. There is no padding or cross-token scaling.
- One E8M0 power-of-two scale per group:
  `2 ** clamp(floor(log2(max(abs(x)))) - 2, -127, 127)`.
  Zero groups use exponent -127. Exponents are extracted directly to avoid
  rounding `log2` near powers of two.
- E2M1 element magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, with signed values,
  round-to-nearest/ties-to-even and saturation at magnitude 6.
- Scale selection and QDQ arithmetic use FP32; the result returns to `dtype`
  before the floating-point Linear operation. Zeros are canonical positive zero.
  A group containing NaN or infinity becomes all NaN. Nonfinite weights are
  rejected during conversion.
- Selected Linear weights undergo QDQ once at initialization. In W4A4, each
  selected Linear's input undergoes QDQ on every forward pass. Bias, embedding,
  normalization, attention matrix products, KV cache, and residual operations
  stay in their original floating precision.

The output embedding/head is always excluded, even if renamed or tied to the
input embedding. Additional exclusions use `ignore`; all aliases of an excluded
module are excluded together. Result JSON includes `config.mxfp4` with the
recipe, requested precision, and exact quantized/excluded module names. Check
this coverage when comparing runs: W4A4 is a Linear-layer policy, not a claim
that every operation in the network uses four bits.

## Supported scope

The initial integration supports dense Transformers **Llama, Qwen2, Qwen3, and
GPT-NeoX** models whose selected layers are ordinary `torch.nn.Linear` modules
with input widths divisible by 32. Model families are explicitly checked rather
than silently accepting an architecture with unquantized custom projections.
`pretraining_tp` must be 1 because older Transformers implementations bypass
Linear forwards when this setting enables tensor-parallel slicing.

CPU arithmetic and tiny-model integration are covered by offline tests. CUDA
and Ascend NPU execution use the same PyTorch operations but need validation on
the target hardware; bitwise equivalence to native FP4 kernels is not implied.
NPU use requires a compatible `torch_npu` installation and `--device npu:0`.
There are no optimized kernels in this backend.

Use one complete floating checkpoint per device. HFLM's data-parallel request
handling is inherited; distributed hardware execution is not covered by the
offline tests. Tensor/model parallelism, Accelerate dispatch/offload, PEFT/delta
application, prequantized checkpoints, specialized Linear subclasses, and hooks
on selected Linear layers are rejected. Merge adapters into a floating
checkpoint beforehand. Shared weight storage across distinct selected modules
is rejected to avoid changing excluded layers. A configuration that selects no
Linear layers fails instead of reporting an unquantized run as W4.

Conversion overwrites the selected model parameters in place and is
inference-only. Reload the original checkpoint for another precision or for
training. A preinitialized model passed through the Python API is also modified
in place. Persistent weight storage is floating point; conversion temporarily
allocates FP32 QDQ intermediates for one layer at a time.
A preinitialized model must already have the requested `dtype`.

## Tests

```bash
python -m pytest tests/models/test_mxfp4.py -q
```

The offline suite compares arithmetic with an independent scalar codebook
oracle, including tie rounding, saturation, scale boundaries, subnormals,
nonfinite groups, dtypes, and batch invariance. It checks conversion safety,
tied output heads, logits for all supported model families, local checkpoint
loading through the registry, the BF16 baseline, and all three Harness request
types. No model or dataset downloads are required by these tests.
