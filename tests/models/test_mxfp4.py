"""Offline arithmetic, conversion and Harness integration tests for MXFP4."""

# Tokenizer special tokens below are vocabulary entries, not credentials.
# ruff: noqa: S106

from __future__ import annotations

import copy
import math

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("accelerate")

from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model
from lm_eval.models.hf_mxfp4 import HFMXFP4
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_mxfp4 import MXFP4Linear, convert_mxfp4, mxfp4_qdq


def reference_qdq(tensor):
    """Scalar codebook oracle, independent of the vectorized rounding algorithm."""
    codebook = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    result = []
    for block in tensor.float().reshape(-1, 32).tolist():
        if not all(math.isfinite(value) for value in block):
            result.extend([math.nan] * 32)
            continue
        maximum = max(map(abs, block))
        exponent = max(-127, min(127, math.frexp(maximum)[1] - 3)) if maximum else -127
        scale = math.ldexp(1.0, exponent)
        for value in block:
            magnitude = abs(value / scale)
            index = min(range(8), key=lambda i: (abs(codebook[i] - magnitude), i % 2))
            rounded = codebook[index] * scale
            result.append(math.copysign(rounded, value) if rounded else 0.0)
    return torch.tensor(result, dtype=tensor.dtype).reshape(tensor.shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_codebook_midpoints_and_saturation(dtype):
    values = [0, 0.25, 0.5, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, 7.5]
    block = torch.tensor((values + [-x for x in values] + [0] * 10), dtype=dtype)
    actual = mxfp4_qdq(block)
    torch.testing.assert_close(actual, reference_qdq(block), rtol=0, atol=0)
    assert actual[:11].tolist() == [0, 0, 0.5, 1, 1, 2, 2, 4, 4, 6, 6]
    assert not torch.signbit(actual[actual == 0]).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_random_blocks_noncontiguous_and_batch_invariance(dtype):
    generator = torch.Generator().manual_seed(13)
    inputs = torch.randn(3, 64, 2, generator=generator).to(dtype).transpose(1, 2)
    assert not inputs.is_contiguous()
    actual = mxfp4_qdq(inputs)
    torch.testing.assert_close(actual, reference_qdq(inputs), rtol=0, atol=0)
    torch.testing.assert_close(
        actual, torch.stack([mxfp4_qdq(row) for row in inputs]), rtol=0, atol=0
    )
    torch.testing.assert_close(actual, mxfp4_qdq(actual), rtol=0, atol=0)
    assert actual.dtype == dtype


def test_scale_boundaries_subnormals_and_nonfinite_blocks():
    powers = torch.tensor([2.0**e for e in (-126, -20, -1, 0, 1, 20, 127)])
    maxima = torch.cat(
        [
            torch.nextafter(powers, torch.zeros_like(powers)),
            powers,
            torch.nextafter(powers, torch.full_like(powers, math.inf)),
            torch.tensor([0.0, 2.0**-149, torch.finfo(torch.float32).max]),
        ]
    )
    inputs = maxima[:, None] * torch.linspace(-1, 1, 32)[None, :]
    inputs = torch.cat([inputs, torch.zeros(3, 32)])
    inputs[-3:, 0] = torch.tensor([math.nan, math.inf, -math.inf])
    actual = mxfp4_qdq(inputs)
    torch.testing.assert_close(
        actual, reference_qdq(inputs), rtol=0, atol=0, equal_nan=True
    )
    assert torch.isnan(actual[-3:]).all()
    assert torch.isfinite(actual[:-3]).all()


@pytest.mark.parametrize("shape", [(), (0,), (31,), (2, 33)])
def test_invalid_block_shape(shape):
    with pytest.raises(ValueError, match="divisible by 32"):
        mxfp4_qdq(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_invalid_dtype(dtype):
    with pytest.raises(TypeError, match="requires float16"):
        mxfp4_qdq(torch.zeros(32, dtype=dtype))


@pytest.mark.parametrize("precision", ["w4a4", "w4a16"])
def test_linear_matches_oracle_and_preserves_parameters(precision):
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Linear(32, 16)).eval()
    original_weight = model[0].weight
    expected_weight = reference_qdq(original_weight)
    original_bias = model[0].bias
    report = convert_mxfp4(model, precision=precision, ignore=[])
    assert report == {"quantized_modules": ["0"], "excluded_modules": []}
    assert model[0].weight is original_weight
    assert model[0].bias is original_bias
    for scale in (0.01, 1, 100):
        inputs = torch.randn(2, 3, 32, requires_grad=True) * scale
        expected_inputs = reference_qdq(inputs) if precision == "w4a4" else inputs
        expected = torch.nn.functional.linear(
            expected_inputs, expected_weight, original_bias
        )
        actual = model(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad
    model.eval()
    with pytest.raises(RuntimeError, match="inference-only"):
        model.train()
    with pytest.raises(ValueError, match="already contains"):
        convert_mxfp4(model, precision=precision, ignore=[])


def test_module_aliases_and_glob_exclusions():
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(32, 32)
    model.alias = model.proj
    model.block = torch.nn.ModuleDict({"router": torch.nn.Linear(32, 2)})
    router_weight = model.block["router"].weight.detach().clone()
    report = convert_mxfp4(model, precision="w4a4", ignore=["*.router"])
    assert model.proj is model.alias
    assert isinstance(model.proj, MXFP4Linear)
    assert report["quantized_modules"] == ["proj", "alias"]
    assert report["excluded_modules"] == ["block.router"]
    torch.testing.assert_close(
        model.block["router"].weight, router_weight, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "failure", ["shape", "shared", "hook", "subclass", "nonfinite", "dispatch", "meta"]
)
def test_incompatible_model_fails_before_changing_weights(failure):
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Linear(32, 32))
    original = model[0].weight.detach().clone()
    if failure == "shape":
        model[1] = torch.nn.Linear(31, 32)
    elif failure == "shared":
        model[1].weight = model[0].weight
    elif failure == "hook":
        model[1].register_forward_pre_hook(lambda *args: None)
    elif failure == "subclass":

        class CustomLinear(torch.nn.Linear):
            pass

        model[1] = CustomLinear(32, 32)
    elif failure == "nonfinite":
        with torch.no_grad():
            model[1].weight[0, 0] = math.nan
    elif failure == "dispatch":
        model[1]._hf_hook = object()
    elif failure == "meta":
        model[1] = torch.nn.Linear(32, 32, device="meta")
    with pytest.raises(ValueError):
        convert_mxfp4(model, precision="w4a4", ignore=[])
    assert type(model[0]) is torch.nn.Linear
    torch.testing.assert_close(model[0].weight, original, rtol=0, atol=0)


def test_empty_coverage_is_not_reported_as_w4():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32))
    with pytest.raises(ValueError, match="No eligible"):
        convert_mxfp4(model, precision="w4a4", ignore=["*"])


def test_excluded_alias_excludes_the_whole_module():
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(32, 32)
    model.alias = model.proj
    model.other = torch.nn.Linear(32, 32)
    report = convert_mxfp4(model, precision="w4a4", ignore=["alias"])
    assert type(model.proj) is torch.nn.Linear
    assert model.proj is model.alias
    assert report == {
        "quantized_modules": ["other"],
        "excluded_modules": ["proj", "alias"],
    }


@pytest.fixture
def tiny_tokenizer():
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[PAD]",
                "[BOS]",
                "[EOS]",
                "[UNK]",
                "hello",
                "world",
                "answer",
                "yes",
                "no",
            ]
        )
    }
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_max_length=64,
    )


def tiny_model(family="Llama"):
    if not hasattr(transformers, f"{family}Config"):
        pytest.skip(f"Transformers version does not provide {family}")
    config = getattr(transformers, f"{family}Config")(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(17)
    return transformers.AutoModelForCausalLM.from_config(config).eval()


@pytest.mark.parametrize("family", ["Llama", "Qwen2", "Qwen3", "GPTNeoX"])
@pytest.mark.parametrize("precision", ["w4a4", "w4a16"])
def test_model_logits_match_independent_reference(family, precision, tiny_tokenizer):
    model = tiny_model(family)
    reference = copy.deepcopy(model)
    original_embedding = model.get_input_embeddings().weight.detach().clone()
    output = reference.get_output_embeddings()
    for module in reference.modules():
        if isinstance(module, torch.nn.Linear) and module is not output:
            with torch.no_grad():
                module.weight.copy_(reference_qdq(module.weight))
            if precision == "w4a4":
                module.register_forward_pre_hook(
                    lambda module, args: (reference_qdq(args[0]),)
                )
    backend = HFMXFP4(
        pretrained=model,
        tokenizer=tiny_tokenizer,
        precision=precision,
        dtype="float32",
        device="cpu",
    )
    tokens = torch.tensor([[1, 4, 5, 6]])
    with torch.no_grad():
        torch.testing.assert_close(
            backend.model(tokens).logits, reference(tokens).logits, rtol=0, atol=0
        )
    torch.testing.assert_close(
        model.get_input_embeddings().weight, original_embedding, rtol=0, atol=0
    )
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    assert not isinstance(model.get_output_embeddings(), MXFP4Linear)
    assert len(backend.quantization_report["quantized_modules"]) > 0


@pytest.mark.parametrize("precision", ["bf16", "w4a16", "w4a4"])
def test_registry_checkpoint_and_all_request_types(tmp_path, tiny_tokenizer, precision):
    model = tiny_model().to(torch.bfloat16)
    model.save_pretrained(tmp_path)
    tiny_tokenizer.save_pretrained(tmp_path)
    backend = get_model("hf-mxfp4").create_from_arg_string(
        f"pretrained={tmp_path.as_posix()},precision={precision},dtype=bfloat16,attn_implementation=eager",
        {"device": "cpu", "batch_size": 2},
    )
    requests = [
        Instance("loglikelihood", {}, (context, " yes"), i)
        for i, context in enumerate(["hello world", "hello"])
    ]
    scores = backend.loglikelihood(requests)
    assert len(scores) == 2 and all(math.isfinite(score) for score, _ in scores)
    rolling = backend.loglikelihood_rolling(
        [Instance("loglikelihood_rolling", {}, ("hello world " * 40,), 0)]
    )
    assert len(rolling) == 1 and math.isfinite(rolling[0])
    generations = backend.generate_until(
        [
            Instance(
                "generate_until",
                {},
                ("hello", {"until": ["[EOS]"], "max_gen_toks": 3, "do_sample": False}),
                0,
            )
        ]
    )
    assert len(generations) == 1 and isinstance(generations[0], str)
    metadata = backend.get_model_info()["mxfp4"]
    assert metadata["precision"] == precision
    assert bool(metadata["quantized_modules"]) == (precision != "bf16")
    if precision == "bf16":
        vanilla = HFLM(
            pretrained=str(tmp_path),
            device="cpu",
            dtype="bfloat16",
            batch_size=2,
            attn_implementation="eager",
        )
        assert scores == vanilla.loglikelihood(requests)


@pytest.mark.parametrize(
    "option",
    ["parallelize", "tp_plan", "peft", "load_in_4bit", "mixed_precision_dtype"],
)
def test_unsupported_options_fail_before_loading(option):
    with pytest.raises(ValueError, match=option):
        HFMXFP4(pretrained="must-not-download", **{option: True})


def test_unsupported_model_and_quantized_config_fail_before_loading():
    config = transformers.GPT2Config()
    with pytest.raises(ValueError, match="supports dense"):
        HFMXFP4._validate_config(config)
    config = transformers.LlamaConfig()
    config.quantization_config = {"quant_method": "bitsandbytes"}
    with pytest.raises(ValueError, match="unquantized"):
        HFMXFP4._validate_config(config)
    config = transformers.LlamaConfig(pretraining_tp=2)
    with pytest.raises(ValueError, match="pretraining_tp=1"):
        HFMXFP4._validate_config(config)


def test_preinitialized_dtype_must_match():
    with pytest.raises(ValueError, match="requested dtype"):
        HFMXFP4(pretrained=tiny_model(), dtype="bfloat16", device="cpu")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"precision": "nvfp4"},
        {"dtype": "auto"},
        {"precision": "bf16", "dtype": "float32"},
    ],
)
def test_invalid_precision_options(kwargs):
    with pytest.raises(ValueError):
        HFMXFP4(pretrained="must-not-download", **kwargs)
