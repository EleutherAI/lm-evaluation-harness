import copy
import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models import api_models, litellm_llms
from lm_eval.models.api_models import JsonChatStr


@pytest.fixture
def sdk(monkeypatch):
    sdk = ModuleType("litellm")
    sdk.completion = Mock()
    sdk.acompletion = AsyncMock()
    monkeypatch.setitem(sys.modules, "litellm", sdk)
    return sdk


@pytest.mark.parametrize("concurrency", [1, 2, 4])
@pytest.mark.parametrize("content", [None, "", "answer"])
def test_generate_until_null_content_matches_sync(sdk, concurrency, content):
    response = Mock()
    response.model_dump.return_value = {
        "choices": [{"index": 0, "message": {"content": content}}]
    }
    sdk.completion.return_value = response
    sdk.acompletion.return_value = response
    model = litellm_llms.LiteLLMChatCompletion(
        model="test-model", num_concurrent=concurrency, eos_string="<eos>"
    )
    model.cache_hook = Mock()
    chat = JsonChatStr('[{"role": "user", "content": "hello"}]')
    options = {"until": ["STOP"]}
    original = copy.deepcopy(options)
    request = Instance("generate_until", {}, (chat, options), 0)

    result = model.generate_until([request], disable_tqdm=True)

    expected = (
        api_models.LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER if content is None else content
    )
    assert result == [expected]
    if concurrency == 1 and content is None:
        model.cache_hook.add_partial.assert_not_called()
    else:
        model.cache_hook.add_partial.assert_called_once_with(
            "generate_until", (chat, options), expected
        )
    assert options == original


@pytest.mark.parametrize("placeholder", ["", "<missing>"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_async_null_replacement_preserves_other_results_and_releases_semaphore(
    monkeypatch, sdk, placeholder, use_cache, caplog
):
    import asyncio

    # The setting is read at module import, just like the shared API backend.
    monkeypatch.setattr(api_models, "LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER", placeholder)
    monkeypatch.setattr(
        litellm_llms, "LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER", placeholder, raising=False
    )
    response = Mock()
    response.model_dump.return_value = {
        "choices": [
            {"index": i, "message": {"content": text}}
            for i, text in enumerate([None, "", "answer"])
        ]
    }
    sdk.acompletion.return_value = response
    model = litellm_llms.LiteLLMChatCompletion(model="test-model", eos_string="<eos>")
    model.cache_hook = Mock()
    keys = ["first", "second", "third"] if use_cache else None

    async def run():
        semaphore = asyncio.Semaphore(1)
        result = await model.amodel_call(
            session=None,
            sem=semaphore,
            messages=(JsonChatStr('[{"role": "user", "content": "hello"}]'),),
            gen_kwargs={},
            cache_keys=keys,
        )
        assert not semaphore.locked()
        return result

    assert asyncio.run(run()) == [placeholder, "", "answer"]
    assert "null content" in caplog.text
    if use_cache:
        assert [call.args for call in model.cache_hook.add_partial.call_args_list] == [
            ("generate_until", key, value)
            for key, value in zip(keys, [placeholder, "", "answer"], strict=True)
        ]
    else:
        model.cache_hook.add_partial.assert_not_called()
