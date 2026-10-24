import copy
import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.litellm_llms import LiteLLMChatCompletion


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize(
    "base_url, overrides, expected_endpoint",
    [
        (None, {}, {}),
        ("http://localhost:4000/v1", {}, {"base_url": "http://localhost:4000/v1"}),
        (
            "http://localhost:4000/v1",
            {"base_url": "http://localhost:5000/v1"},
            {"base_url": "http://localhost:5000/v1"},
        ),
        (
            "http://localhost:4000/v1",
            {"api_base": "http://localhost:5000/v1"},
            {"api_base": "http://localhost:5000/v1"},
        ),
        (
            None,
            {"base_url": "http://localhost:5000/v1"},
            {"base_url": "http://localhost:5000/v1"},
        ),
        (
            None,
            {"api_base": "http://localhost:5000/v1"},
            {"api_base": "http://localhost:5000/v1"},
        ),
    ],
)
def test_generate_until_forwards_configured_endpoint(
    monkeypatch, concurrency, base_url, overrides, expected_endpoint
):
    # Exercise the harness/SDK boundary without installing LiteLLM or calling a provider.
    sdk = ModuleType("litellm")
    response = Mock()
    response.model_dump.return_value = {
        "choices": [{"index": 0, "message": {"content": "answer"}}]
    }
    sdk.completion = Mock(return_value=response)
    sdk.acompletion = AsyncMock(return_value=response)
    monkeypatch.setitem(sys.modules, "litellm", sdk)
    model = LiteLLMChatCompletion(
        model="openai/test-model",
        base_url=base_url,
        num_concurrent=concurrency,
        eos_string="<eos>",
    )
    chat = JsonChatStr('[{"role": "user", "content": "hello"}]')
    options = {"max_gen_toks": 17, "until": ["STOP"], **overrides}
    original = copy.deepcopy(options)
    request = Instance("generate_until", {}, (chat, options), 0)

    assert model.generate_until([request], disable_tqdm=True) == ["answer"]

    call = sdk.completion if concurrency == 1 else sdk.acompletion
    call.assert_called_once()
    payload = call.call_args.kwargs
    assert {
        k: payload[k] for k in ("base_url", "api_base") if k in payload
    } == expected_endpoint
    assert payload["model"] == "openai/test-model"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 17
    assert options == original
