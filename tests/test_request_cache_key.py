"""Tests for the request-cache key built by `Task.request_cache_key`.

The key names the file that `--cache_requests` reads back. Any run setting that
changes the resulting Instances has to be part of it, otherwise a later run
silently reuses instances that were built under different settings.
"""

from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import TaskConfig


def make_task(**overrides) -> ConfigurableTask:
    """Build a `generate_until` task without touching the network or datasets."""
    config = {
        "task": "test_task",
        "output_type": "generate_until",
        "doc_to_text": "Question: {{question}}",
        "doc_to_target": "{{answer}}",
        "num_fewshot": 5,
        "generation_kwargs": {"until": ["\n"], "do_sample": False, "temperature": 0.0},
    }
    config.update(overrides)

    task = ConfigurableTask.__new__(ConfigurableTask)
    task._config = TaskConfig(**config)
    task.OUTPUT_TYPE = config["output_type"]
    return task


RUN_CONFIG = {
    "rank": 0,
    "world_size": 1,
    "system_instruction": None,
    "apply_chat_template": False,
    "fewshot_as_multiturn": False,
    "tokenizer_name": "meta-llama/Llama-3-8B",
}


def test_generation_kwargs_change_the_cache_key():
    """Two runs differing only in sampling parameters must not share a cache entry.

    `generate_until` bakes `generation_kwargs` into each Instance's arguments, so
    reusing the greedy run's cache for a temperature=0.7 run would silently evaluate
    at the old sampling settings.
    """
    greedy = make_task(
        generation_kwargs={"until": ["\n"], "do_sample": False, "temperature": 0.0}
    )
    sampled = make_task(
        generation_kwargs={"until": ["\n"], "do_sample": True, "temperature": 0.7}
    )

    assert greedy.request_cache_key(**RUN_CONFIG) != sampled.request_cache_key(
        **RUN_CONFIG
    )


def test_max_gen_toks_and_until_change_the_cache_key():
    """`until` and `max_gen_toks` also reach the model, so they belong in the key."""
    base = make_task(generation_kwargs={"until": ["\n"], "max_gen_toks": 30})
    longer = make_task(generation_kwargs={"until": ["\n"], "max_gen_toks": 256})
    other_stop = make_task(generation_kwargs={"until": ["\n\n"], "max_gen_toks": 30})

    keys = {
        base.request_cache_key(**RUN_CONFIG),
        longer.request_cache_key(**RUN_CONFIG),
        other_stop.request_cache_key(**RUN_CONFIG),
    }
    assert len(keys) == 3


def test_cache_key_is_stable_across_key_order():
    """The same kwargs written in a different order describe the same run."""
    one = make_task(
        generation_kwargs={"until": ["\n"], "do_sample": True, "temperature": 0.7}
    )
    two = make_task(
        generation_kwargs={"temperature": 0.7, "until": ["\n"], "do_sample": True}
    )

    assert one.request_cache_key(**RUN_CONFIG) == two.request_cache_key(**RUN_CONFIG)


def test_cache_key_is_deterministic():
    """Repeated calls on the same task give the same key."""
    task = make_task()
    assert task.request_cache_key(**RUN_CONFIG) == task.request_cache_key(**RUN_CONFIG)


def test_cache_key_unchanged_when_generation_kwargs_unset():
    """Tasks without `generation_kwargs` keep their previous key, so caches stay valid."""
    task = make_task(output_type="multiple_choice", generation_kwargs=None)
    key = task.request_cache_key(**RUN_CONFIG)

    assert key == (
        "requests-test_task-5shot-rank0-world_size1-tokenizermeta-llama/Llama-3-8B"
    )
    assert "gen_kwargs_hash" not in key


def test_existing_run_settings_still_change_the_cache_key():
    """The settings the key already covered must keep working."""
    task = make_task()
    base = task.request_cache_key(**RUN_CONFIG)

    assert task.request_cache_key(**{**RUN_CONFIG, "rank": 1}) != base
    assert task.request_cache_key(**{**RUN_CONFIG, "world_size": 2}) != base
    assert task.request_cache_key(**{**RUN_CONFIG, "apply_chat_template": True}) != base
    assert (
        task.request_cache_key(**{**RUN_CONFIG, "fewshot_as_multiturn": True}) != base
    )
    assert (
        task.request_cache_key(**{**RUN_CONFIG, "system_instruction": "be terse"})
        != base
    )
    assert task.request_cache_key(**{**RUN_CONFIG, "tokenizer_name": "gpt2"}) != base
