"""Regression tests for the request-cache key (issue #3881).

The request cache key must incorporate ``generation_kwargs`` so that changing a
sampling parameter (temperature, top_p, max_gen_toks, until, ...) between two
``--cache_requests`` runs produces a cache miss instead of silently reusing the
instances that were built for the previous sampling settings.
"""

from lm_eval.api.task import Task


BASE = {
    "task": "gsm8k",
    "num_fewshot": 5,
    "rank": 0,
    "world_size": 1,
    "apply_chat_template": False,
    "fewshot_as_multiturn": False,
    "system_instruction": None,
    "tokenizer_name": "meta-llama/Llama-3-8B",
}


def key(**overrides):
    return Task._build_request_cache_key(**{**BASE, **overrides})


def test_temperature_change_produces_different_key():
    greedy = key(generation_kwargs={"until": ["\n\n"], "do_sample": False})
    sampled = key(
        generation_kwargs={"until": ["\n\n"], "do_sample": True, "temperature": 0.7}
    )
    assert greedy != sampled


def test_max_gen_toks_change_produces_different_key():
    short = key(generation_kwargs={"until": ["\n\n"], "max_gen_toks": 64})
    long = key(generation_kwargs={"until": ["\n\n"], "max_gen_toks": 256})
    assert short != long


def test_identical_generation_kwargs_produce_identical_key():
    gk = {"until": ["\n\n"], "temperature": 0.7, "top_p": 0.95}
    assert key(generation_kwargs=gk) == key(generation_kwargs=dict(gk))


def test_key_is_insensitive_to_dict_order():
    a = key(generation_kwargs={"temperature": 0.7, "top_p": 0.95, "until": ["\n\n"]})
    b = key(generation_kwargs={"until": ["\n\n"], "top_p": 0.95, "temperature": 0.7})
    assert a == b


def test_multiple_choice_key_has_no_gen_kwargs_suffix():
    # multiple_choice tasks carry no generation_kwargs; their key must stay
    # unchanged by this fix (backward compatibility / no spurious cache miss).
    mc = key(generation_kwargs=None)
    assert "gen_kwargs" not in mc
    assert mc.endswith("-tokenizermeta-llama/Llama-3-8B")


def test_run_config_components_still_reflected():
    # the pre-existing run-config components must still change the key
    assert key(rank=0) != key(rank=1)
    assert key(tokenizer_name="a") != key(tokenizer_name="b")
    # with no generation_kwargs the key is the original (suffix-free) form
    assert "gen_kwargs" not in key()
