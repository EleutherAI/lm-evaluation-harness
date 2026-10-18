import random
import types

import pytest

import lm_eval.api.metrics as metrics
from lm_eval.api.task import Task


def test_bootstrapping():
    random.seed(42)
    arr = [random.random() for _ in range(1000)]
    expected = metrics.mean_stderr(arr)
    bootstrapped = metrics.bootstrap_stderr(metrics.mean, arr, iters=100000)

    assert bootstrapped == pytest.approx(expected, abs=1e-4)


def _doc_iterator_stub(docs):
    """Minimal stand-in for a Task, exercising the real Task.doc_iterator."""
    return types.SimpleNamespace(
        eval_docs=docs,
        config=types.SimpleNamespace(task="stub_task"),
    )


def test_doc_iterator_empty_samples_selects_nothing():
    # samples=[] is an explicit, empty selection: it must yield no documents.
    # Regression for `if samples:` treating [] like None (full-dataset path).
    stub = _doc_iterator_stub(["doc0", "doc1", "doc2"])
    assert list(Task.doc_iterator(stub, samples=[])) == []


def test_doc_iterator_none_samples_selects_all():
    # samples=None means "no explicit selection": iterate the full dataset.
    stub = _doc_iterator_stub(["doc0", "doc1", "doc2"])
    assert list(Task.doc_iterator(stub, samples=None)) == [
        (0, "doc0"),
        (1, "doc1"),
        (2, "doc2"),
    ]


def test_doc_iterator_explicit_samples_selects_subset():
    stub = _doc_iterator_stub(["doc0", "doc1", "doc2"])
    assert list(Task.doc_iterator(stub, samples=[2])) == [(0, "doc2")]
