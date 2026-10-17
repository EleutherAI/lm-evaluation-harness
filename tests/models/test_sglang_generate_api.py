"""Offline tests for the SGLang generate API backend.

Covers payload assembly (`_create_payload`) and generation response parsing
(`parse_generations`) of `lm_eval.models.sglang_generate_API.SGLANGGENERATEAPI`
without a running SGLang server or GPU, as requested in #1613.
"""

import pytest

from lm_eval.models.sglang_generate_API import SGLANGGENERATEAPI


@pytest.fixture
def model():
    """Minimal instance: payload assembly only reads `_max_gen_toks` from state."""
    instance = SGLANGGENERATEAPI.__new__(SGLANGGENERATEAPI)
    instance._max_gen_toks = 256
    return instance


class TestCreatePayloadGenerate:
    """Payload assembly for the generate path (`generate=True`)."""

    def test_string_input_maps_to_text(self, model):
        payload = model._create_payload("hello", generate=True, gen_kwargs={})
        assert payload["text"] == "hello"
        assert "input_ids" not in payload

    def test_token_input_maps_to_input_ids(self, model):
        payload = model._create_payload([[1, 2, 3]], generate=True, gen_kwargs={})
        assert payload["input_ids"] == [[1, 2, 3]]
        assert "text" not in payload

    def test_max_tokens_takes_precedence_over_max_gen_toks(self, model):
        payload = model._create_payload(
            [[1]],
            generate=True,
            gen_kwargs={"max_tokens": 16, "max_gen_toks": 999},
        )
        assert payload["sampling_params"]["max_new_tokens"] == 16

    def test_max_gen_toks_falls_back_to_instance_default(self, model):
        payload = model._create_payload([[1]], generate=True, gen_kwargs={})
        assert payload["sampling_params"]["max_new_tokens"] == 256

    def test_temperature_defaults_to_zero(self, model):
        payload = model._create_payload([[1]], generate=True, gen_kwargs={})
        assert payload["sampling_params"]["temperature"] == 0

    def test_do_sample_is_dropped(self, model):
        payload = model._create_payload(
            [[1]], generate=True, gen_kwargs={"do_sample": False}
        )
        assert "do_sample" not in payload["sampling_params"]

    def test_until_and_eos_merge_into_stop(self, model):
        payload = model._create_payload(
            [[1]], generate=True, gen_kwargs={"until": ["</s>"]}, eos="[EOS]"
        )
        assert payload["sampling_params"]["stop"] == ["</s>", "[EOS]"]

    def test_until_none_yields_empty_stop(self, model):
        payload = model._create_payload([[1]], generate=True, gen_kwargs={})
        assert payload["sampling_params"]["stop"] == []

    def test_extra_gen_kwargs_pass_through(self, model):
        payload = model._create_payload([[1]], generate=True, gen_kwargs={"top_p": 0.9})
        assert payload["sampling_params"]["top_p"] == 0.9


class TestCreatePayloadLoglikelihood:
    """Payload assembly for the loglikelihood path (`generate=False`)."""

    def test_rejects_string_input(self, model):
        with pytest.raises(AssertionError):
            model._create_payload("hello", generate=False, gen_kwargs={})

    def test_required_fields(self, model):
        payload = model._create_payload([[1, 2]], generate=False, gen_kwargs={})
        assert payload["input_ids"] == [[1, 2]]
        sampling_params = payload["sampling_params"]
        assert sampling_params["max_new_tokens"] == 1
        assert sampling_params["temperature"] == 0
        assert payload["logprob_start_len"] == 0
        assert payload["top_logprobs_num"] == 1
        assert payload["return_logprob"] is True


class TestParseGenerations:
    """Generation response parsing (`parse_generations`)."""

    def test_single_dict_is_wrapped_in_list(self):
        assert SGLANGGENERATEAPI.parse_generations({"text": "hi"}) == ["hi"]

    def test_list_preserves_order_and_whitespace(self):
        outputs = [{"text": " a "}, {"text": "b"}]
        assert SGLANGGENERATEAPI.parse_generations(outputs) == [" a ", "b"]
