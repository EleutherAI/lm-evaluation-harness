"""Request-cache isolation for API models with local chat templates."""

import json

import pytest
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from lm_eval.api.task import ConfigurableTask
from lm_eval.caching import cache
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


class CacheTask(ConfigurableTask):
    def download(self, dataset_kwargs=None, **kwargs):
        self.dataset = DatasetDict(
            {
                "test": Dataset.from_list(
                    [{"question": "A question", "answer": "An answer"}]
                )
            }
        )


def make_task():
    return CacheTask(
        config={
            "task": "api-cache-identity",
            "test_split": "test",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "output_type": "generate_until",
            "metric_list": [{"metric": "bypass"}],
            "generation_kwargs": {"until": [], "max_gen_toks": 1},
        }
    )


def build_prompt(model):
    task = make_task()
    task.build_all_requests(
        cache_requests=True,
        apply_chat_template=True,
        chat_template=model.apply_chat_template,
        tokenizer_name=model.tokenizer_name,
    )
    return task.instances[0].args[0]


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path / "request-cache"))
    result = []
    unknown = "[UNK]"
    for name in ("first", "second"):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({unknown: 0}, unk_token=unknown)),
            unk_token=unknown,
            chat_template=(
                name
                + "{% for message in messages %}|{{ message['role'] }}:{{ message['content'] }}{% endfor %}"
                + "{% if add_generation_prompt %}|assistant:{% endif %}"
            ),
        )
        # Absolute paths exercise Windows drive letters as well as directory separators.
        directory = tmp_path / name
        tokenizer.save_pretrained(directory)
        result.append(
            LocalCompletionsAPI(
                model="same-served-model",
                base_url="http://localhost/v1/completions",
                tokenizer=str(directory),
                tokenizer_backend="huggingface",
            )
        )
    return result


def test_request_cache_uses_each_hf_tokenizer_template(models, monkeypatch):
    first, second = models
    assert build_prompt(first) == "first|user:A question|assistant:"
    assert build_prompt(second) == "second|user:A question|assistant:"

    def unexpected_render(*args, **kwargs):
        pytest.fail("The same tokenizer should reuse its cached requests")

    monkeypatch.setattr(second, "apply_chat_template", unexpected_render)
    assert build_prompt(second) == "second|user:A question|assistant:"
    assert build_prompt(first) == "first|user:A question|assistant:"


@pytest.mark.parametrize("chat_first", [False, True])
def test_request_cache_separates_rendered_and_structured_chats(models, chat_first):
    rendered = models[0]
    chat = LocalChatCompletion(
        model="same-served-model", base_url="http://localhost/v1/chat/completions"
    )
    for model in (chat, rendered) if chat_first else (rendered, chat):
        prompt = build_prompt(model)
        if model is rendered:
            assert prompt == "first|user:A question|assistant:"
        else:
            assert isinstance(prompt, JsonChatStr)
            assert json.loads(prompt.prompt) == [
                {"role": "user", "content": "A question"}
            ]


def test_unrendered_hf_chat_keeps_structured_request_cache(models, monkeypatch):
    model = models[0]
    model.tokenized_requests = False
    chat = LocalChatCompletion(
        model="another-served-model", base_url="http://localhost/v1/chat/completions"
    )
    prompt = build_prompt(model)

    def unexpected_render(*args, **kwargs):
        pytest.fail("Structured chat requests can reuse the same cached messages")

    monkeypatch.setattr(chat, "apply_chat_template", unexpected_render)
    assert build_prompt(chat) == prompt
