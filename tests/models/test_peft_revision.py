"""Offline checks that base models and PEFT adapters use independent revisions."""

from types import SimpleNamespace
from unittest.mock import Mock

import peft
import pytest
import tokenizers
import transformers

from lm_eval.models.huggingface import HFLM


@pytest.fixture(scope="module")
def local_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("peft-base")
    config = transformers.GPT2Config(
        vocab_size=4,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )
    transformers.GPT2LMHeadModel(config).save_pretrained(path)
    bos, eos, unknown = "<bos>", "<eos>", "<unk>"
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel(
                {"<bos>": 0, "<eos>": 1, "<unk>": 2, "word": 3}, unk_token=unknown
            )
        ),
        bos_token=bos,
        eos_token=eos,
        unk_token=unknown,
        pad_token=eos,
    )
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("adapter_revision", [None, "adapter-checkpoint", "main"])
def test_peft_revision_loads_and_reports_independently(
    local_model,
    adapter_revision,
    monkeypatch,
):
    adapter_loader = Mock(side_effect=lambda model, *args, **kwargs: model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", adapter_loader)
    base_loader = Mock(wraps=transformers.AutoModelForCausalLM.from_pretrained)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", base_loader
    )
    tokenizer_loader = Mock(wraps=transformers.AutoTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    config_loader = Mock(wraps=transformers.AutoConfig.from_pretrained)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config_loader)
    hub_info = Mock(
        side_effect=lambda **kwargs: SimpleNamespace(sha=kwargs["revision"] + "-sha")
    )
    monkeypatch.setattr(
        "lm_eval.models.huggingface.HfApi", lambda: SimpleNamespace(model_info=hub_info)
    )

    model = HFLM(
        pretrained=local_model,
        revision="base-checkpoint",
        peft="example/adapter",
        peft_revision=adapter_revision,
        device="cpu",
        dtype="float32",
    )
    expected_revision = adapter_revision or "base-checkpoint"
    assert adapter_loader.call_args.args[1] == "example/adapter"
    assert adapter_loader.call_args.kwargs == {"revision": expected_revision}
    for loader in [base_loader, tokenizer_loader, config_loader]:
        assert loader.call_args.kwargs["revision"] == "base-checkpoint"
        assert "peft_revision" not in loader.call_args.kwargs
    info = model.get_model_info()
    assert info["model_revision"] == "base-checkpoint"
    assert info["model_sha"] == "base-checkpoint-sha"
    assert info["peft_revision"] == expected_revision
    assert info["peft_sha"] == expected_revision + "-sha"
    assert hub_info.call_args.kwargs == {
        "repo_id": "example/adapter",
        "revision": expected_revision,
    }
