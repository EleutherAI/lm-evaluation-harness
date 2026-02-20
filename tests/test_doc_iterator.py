"""Verify doc_iterator yields real doc IDs (not re-enumerated 0,1,2,...)."""

from unittest.mock import PropertyMock, patch

import pytest

from lm_eval.api.task import ConfigurableTask


DOCS = ["a", "b", "c", "d", "e", "f", "g", "h"]


@pytest.fixture()
def task():
    task = ConfigurableTask.__new__(ConfigurableTask)
    task._config = type("C", (), {"task": "test"})()
    return task


def _ids(task, **kwargs):
    with patch.object(
        type(task), "eval_docs", new_callable=PropertyMock, return_value=DOCS
    ):
        return list(task.doc_iterator(**kwargs))


def test_all_docs(task):
    assert _ids(task) == [
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (3, "d"),
        (4, "e"),
        (5, "f"),
        (6, "g"),
        (7, "h"),
    ]


def test_limit(task):
    assert _ids(task, limit=3) == [(0, "a"), (1, "b"), (2, "c")]


def test_rank_and_world_size(task):
    # rank 0 gets even indices, rank 1 gets odd
    assert _ids(task, rank=0, world_size=2) == [
        (0, "a"),
        (2, "c"),
        (4, "e"),
        (6, "g"),
    ]
    assert _ids(task, rank=1, world_size=2) == [
        (1, "b"),
        (3, "d"),
        (5, "f"),
        (7, "h"),
    ]


def test_samples_preserves_real_ids(task):
    assert _ids(task, samples=[2, 5, 7]) == [(2, "c"), (5, "f"), (7, "h")]


def test_samples_with_rank(task):
    # two workers splitting 3 samples: rank 0 gets [2,7], rank 1 gets [5]
    assert _ids(task, samples=[2, 5, 7], rank=0, world_size=2) == [(2, "c"), (7, "h")]
    assert _ids(task, samples=[2, 5, 7], rank=1, world_size=2) == [(5, "f")]
