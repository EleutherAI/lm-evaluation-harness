import types

import pytest

from lm_eval.evaluator import _generation_length_stats, _iter_generations, _percentile
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.utils import make_generation_length_table


class FakeTokenizer:
    """Whitespace tokenizer, so a length is the number of words."""

    def encode(self, text, add_special_tokens=False, truncation=False):
        return text.split()


def make_lm(tokenizer=True):
    # bypass TemplateAPI.__init__, which would load a tokenizer from the network
    lm = object.__new__(LocalCompletionsAPI)
    lm.tokenizer = FakeTokenizer() if tokenizer else None
    lm.tokenizer_backend = "huggingface" if tokenizer else "remote"
    return lm


def make_acc(output_type="generate_until", samples=None):
    task = types.SimpleNamespace(OUTPUT_TYPE=output_type)
    return {"task": task, "logged_samples": samples or []}


GEN_SAMPLE = {
    "doc_id": 0,
    "filter": "flexible-extract",
    "resps": [["one two"]],
    "full_resps": [["one two three four five"]],
}


def test_generation_length_prefers_full_resps():
    assert list(_iter_generations(GEN_SAMPLE)) == ["one two three four five"]
    assert list(_iter_generations({"resps": [["a b"]]})) == ["a b"]
    assert list(_iter_generations({"resps": ["a", "b"]})) == ["a", "b"]
    assert list(_iter_generations({"resps": [["a", "b"], ["c"]]})) == ["a", "b", "c"]


def test_generation_length_rejects_unexpected_shape():
    with pytest.raises(ValueError, match="unexpected resps shape"):
        list(_iter_generations({"doc_id": 3, "resps": [[[1.0, True]]]}))


def test_generation_length_counts_each_document_once():
    # the same document is logged once per filter
    samples = [GEN_SAMPLE, {**GEN_SAMPLE, "filter": "strict-match"}]
    stats = _generation_length_stats(
        make_lm(), {"gsm8k_platinum": make_acc(samples=samples)}
    )
    assert stats["gsm8k_platinum"]["count"] == 1
    assert stats["gsm8k_platinum"]["mean"] == 5.0


def test_generation_length_is_na_for_non_generative_tasks():
    stats = _generation_length_stats(
        make_lm(),
        {
            "gsm8k_platinum": make_acc(samples=[GEN_SAMPLE]),
            "mmlu": make_acc(output_type="loglikelihood"),
        },
    )
    assert stats["gsm8k_platinum"]["count"] == 1
    assert stats["mmlu"] == {
        "count": 0,
        "mean": None,
        "p25": None,
        "median": None,
        "p75": None,
        "p90": None,
    }


def test_generation_length_table_renders_na():
    table = make_generation_length_table(
        {
            "gsm8k_platinum": {
                "count": 500,
                "mean": 197.078,
                "p25": 122.75,
                "median": 183.0,
                "p75": 263.25,
                "p90": 365.1,
            },
            "mmlu": {
                "count": 0,
                "mean": None,
                "p25": None,
                "median": None,
                "p75": None,
                "p90": None,
            },
        }
    )
    assert table.splitlines() == [
        "|Tasks         |n  |p25   |mean  |median|p75   |p90   |",
        "|--------------|---|------|------|------|------|------|",
        "|gsm8k_platinum|500|122.75|197.08|183.00|263.25|365.10|",
        "|mmlu          |N/A|N/A   |N/A   |N/A   |N/A   |N/A   |",
    ]


def test_percentile_interpolates():
    assert _percentile([1, 2, 3, 4], 50) == 2.5
    assert _percentile([10], 90) == 10.0
    assert _percentile([0, 10], 25) == 2.5


class _NativeBackend:
    """Stand-in for a TemplateLM that is not an OpenAI-compatible client."""

    OUTPUT_TYPE = "generate_until"
    tokenizer = FakeTokenizer()


def test_generation_length_is_not_implemented_for_native_backends():
    for lm in (object(), _NativeBackend()):
        with pytest.raises(NotImplementedError, match="is not implemented for"):
            _generation_length_stats(lm, {"t": make_acc()})


def test_generation_length_requires_tokenizer():
    with pytest.raises(ValueError, match="requires a local tokenizer"):
        _generation_length_stats(make_lm(tokenizer=False), {"t": make_acc()})


def test_generation_length_requires_samples():
    with pytest.raises(ValueError, match="requires --log_samples"):
        _generation_length_stats(make_lm(), {})
