"""Tests for the render cache on ConfigurableTask.doc_to_text / doc_to_target /
doc_to_choice.

The cache key is `(kind, id(doc), id(template))`. A repeat call with the same
doc and same template should hit cache on the second invocation; different
docs or different templates should not collide.
"""

from __future__ import annotations

from unittest.mock import Mock

from lm_eval.api.task import ConfigurableTask


def _make_task():
    """Build a minimal ConfigurableTask instance without running __init__.

    We only need the cache wrapper logic to be exercised; the underlying
    `_doc_to_*_render` helpers are stubbed so we can count calls.
    """
    task = ConfigurableTask.__new__(ConfigurableTask)
    task._render_cache = {}
    task.prompt = None

    cfg = Mock()
    cfg.doc_to_text = "{{q}}"
    cfg.doc_to_target = "{{a}}"
    cfg.doc_to_choice = "{{c}}"
    task.config = cfg
    task._config = cfg
    task.features = []

    return task


def test_doc_to_text_hits_cache_on_repeat_doc():
    task = _make_task()
    task._doc_to_text_render = Mock(return_value="rendered")

    doc = {"q": "test"}
    r1 = task.doc_to_text(doc)
    r2 = task.doc_to_text(doc)

    assert r1 == r2 == "rendered"
    assert task._doc_to_text_render.call_count == 1, (
        "second call with the same doc should hit cache"
    )


def test_doc_to_text_different_docs_dont_collide():
    task = _make_task()
    task._doc_to_text_render = Mock(side_effect=lambda doc, _: f"rendered:{doc['q']}")

    doc_a = {"q": "first"}
    doc_b = {"q": "second"}
    r_a1 = task.doc_to_text(doc_a)
    r_b = task.doc_to_text(doc_b)
    r_a2 = task.doc_to_text(doc_a)  # cache hit for doc_a

    assert r_a1 == "rendered:first"
    assert r_b == "rendered:second"
    assert r_a2 == "rendered:first"
    # 2 unique docs => 2 renders, third call hits cache
    assert task._doc_to_text_render.call_count == 2


def test_doc_to_target_hits_cache_on_repeat_doc():
    task = _make_task()
    task._doc_to_target_render = Mock(return_value="answer")

    doc = {"a": "test"}
    r1 = task.doc_to_target(doc)
    r2 = task.doc_to_target(doc)

    assert r1 == r2 == "answer"
    assert task._doc_to_target_render.call_count == 1


def test_doc_to_choice_hits_cache_on_repeat_doc():
    task = _make_task()
    task.config.doc_to_choice = "{{c}}"  # not None, so we hit the else branch
    task._doc_to_choice_render = Mock(return_value=["a", "b"])

    doc = {"c": "test"}
    r1 = task.doc_to_choice(doc)
    r2 = task.doc_to_choice(doc)

    assert r1 == r2 == ["a", "b"]
    assert task._doc_to_choice_render.call_count == 1


def test_falls_through_when_render_cache_missing():
    """A subclass that bypasses ConfigurableTask.__init__ won't have
    `_render_cache`; the wrapper should still work, just without caching."""
    task = _make_task()
    del task._render_cache
    task._doc_to_text_render = Mock(return_value="rendered")

    doc = {"q": "test"}
    r1 = task.doc_to_text(doc)
    r2 = task.doc_to_text(doc)

    assert r1 == r2 == "rendered"
    # Without a cache, both calls render fresh.
    assert task._doc_to_text_render.call_count == 2
