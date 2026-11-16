"""Tests for the RULER synthetic tasks.

Two independent things are covered here.

**The samplers.** Every sampler sizes its prompts by shrinking the context
until it fits `max_seq_length`. When the smallest context a sample can have
still overflows, the shrinking loops used to stop making progress and spin
forever instead of giving up on the sample.

**Tokenizer resolution.** These tasks build their own data, so they are handed
the parsed `--model_args` and have to find a model name in it. Nothing in that
half downloads a dataset or loads a tokenizer: every case is one where
resolution fails, which is exactly the path that used to crash.

`cwe_utils` needs `wonderwords` from the `ruler` extra, which the unit-test
workflow does not install; that builder is skipped rather than the whole
module, so the rest still run in CI.
"""

import importlib

import pytest

from lm_eval.tasks.ruler.common_utils import get_tokenizer, resolve_tokenizer_name
from lm_eval.tasks.ruler.qa_utils import generate_samples


class RunawayLoop(BaseException):
    """Signals a non-terminating shrink loop.

    Deliberately a `BaseException`: the samplers treat any `Exception` raised
    while building a sample as "try a smaller context", so an ordinary error
    would be swallowed by the very loop under test.
    """


class FakeTokenizer:
    """Tokenizer stand-in: one token per whitespace-separated word.

    Raises once it has been called far more often than a terminating run needs,
    so a regression fails the test instead of hanging the suite.
    """

    class _Encoded:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    def __init__(self, max_calls: int = 1000):
        self.max_calls = max_calls
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RunawayLoop(f"tokenizer called more than {self.max_calls} times")
        return self._Encoded(text.split())


def make_qa_corpus(doc_words: int, gold_docs: int, num_qas: int = 20):
    """A corpus where every question needs `gold_docs` documents of fixed size."""
    docs = [" ".join([f"w{i}"] * doc_words) for i in range(200)]
    qas = [
        {
            "query": f"question {i}?",
            "outputs": [f"answer{i}"],
            # HotpotQA-style: each question is supported by several gold
            # documents and carries no `more_context` distractors.
            "context": [(i + j) % len(docs) for j in range(gold_docs)],
        }
        for i in range(num_qas)
    ]
    return docs, qas


def test_generate_samples_rejects_impossible_seq_length():
    """When nothing fits, fail with an actionable error instead of looping."""
    docs, qas = make_qa_corpus(doc_words=100, gold_docs=10)
    # 10 gold docs of 100 words each is far past this, so no amount of
    # shrinking makes any question fit.
    with pytest.raises(ValueError, match="Could not generate any QA sample"):
        generate_samples(
            tokenizer=FakeTokenizer(),
            docs=docs,
            qas=qas,
            max_seq_length=200,
            num_samples=5,
            tokens_to_generate=32,
            incremental=10,
        )


def test_generate_samples_keeps_questions_that_fit():
    """Samples that fit are still produced, and stay within the budget."""
    docs, qas = make_qa_corpus(doc_words=5, gold_docs=2)
    samples = generate_samples(
        tokenizer=FakeTokenizer(),
        docs=docs,
        qas=qas,
        max_seq_length=4096,
        num_samples=5,
        tokens_to_generate=32,
        incremental=10,
    )
    assert len(samples) == 5
    assert all(s["length"] <= 4096 for s in samples)
    assert all(s["max_length"] == 4096 for s in samples)
    # The answer has to survive into the sample.
    assert [s["outputs"] for s in samples] == [q["outputs"] for q in qas[:5]]


def test_generate_samples_drops_only_the_oversized():
    """A corpus where only some questions fit yields the ones that do."""
    docs, qas = make_qa_corpus(doc_words=5, gold_docs=2)
    # Make one question impossible by giving it many more gold documents.
    qas[2]["context"] = list(range(60))
    samples = generate_samples(
        tokenizer=FakeTokenizer(),
        docs=docs,
        qas=qas,
        max_seq_length=120,
        num_samples=5,
        tokens_to_generate=32,
        incremental=10,
    )
    assert [s["index"] for s in samples] == [0, 1, 3, 4]


# Every task builder reachable from a `custom_dataset: !function ...` entry in
# lm_eval/tasks/ruler/*.yaml, as a pytest param so that a builder whose
# optional dependency is absent skips on its own.
BUILDER_SPECS = [
    ("qa_utils", ("get_squad", "get_hotpotqa")),
    ("cwe_utils", ("get_cw_dataset",)),
    ("fwe_utils", ("fwe_download",)),
    ("vt_utils", ("get_vt_dataset",)),
    (
        "niah_utils",
        (
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multivalue",
            "niah_multiquery",
        ),
    ),
]


def _builder_params():
    params = []
    for module_name, builder_names in BUILDER_SPECS:
        try:
            module = importlib.import_module(f"lm_eval.tasks.ruler.{module_name}")
        except ImportError as exc:
            params.extend(
                pytest.param(
                    None,
                    id=name,
                    marks=pytest.mark.skip(reason=f"{module_name}: {exc}"),
                )
                for name in builder_names
            )
            continue
        params.extend(
            pytest.param(getattr(module, name), id=name) for name in builder_names
        )
    return params


BUILDERS = _builder_params()


class TestResolveTokenizerName:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"tokenizer": "tok"}, "tok"),
            ({"pretrained": "pre"}, "pre"),
            ({"model": "mod"}, "mod"),
            # `tokenizer` wins over `pretrained`, which wins over `model`.
            ({"tokenizer": "tok", "pretrained": "pre", "model": "mod"}, "tok"),
            ({"pretrained": "pre", "model": "mod"}, "pre"),
            # An API backend names the model `model` and adds its own args.
            ({"model": "org/name", "base_url": "http://localhost:8080/v1"}, "org/name"),
        ],
    )
    def test_finds_the_model_name(self, kwargs, expected):
        assert resolve_tokenizer_name(kwargs) == expected

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"base_url": "http://localhost:8080/v1"},
            # Empty and non-string values are not usable names. A dict is the
            # one that used to be passed through as a default.
            {"tokenizer": ""},
            {"pretrained": {}},
            {"model": None},
            {"tokenizer": ["a", "b"]},
        ],
    )
    def test_returns_none_when_there_is_no_usable_name(self, kwargs):
        assert resolve_tokenizer_name(kwargs) is None

    def test_result_is_hashable_so_the_cached_lookup_can_accept_it(self):
        # get_tokenizer is decorated with functools.cache, which hashes its
        # arguments before the body runs.
        hash(resolve_tokenizer_name({"pretrained": {}}))


class TestGetTokenizer:
    def test_missing_name_reaches_the_assertion(self):
        with pytest.raises(AssertionError, match="No tokenizer or pretrained"):
            get_tokenizer(None)

    def test_the_message_names_the_argument_that_fixes_it(self):
        with pytest.raises(AssertionError, match=r"--model_args tokenizer="):
            get_tokenizer(None)


class TestBuildersWithoutATokenizer:
    """A missing tokenizer must be reported, not crashed on.

    Before the fix, `qa_utils`, `cwe_utils` and `fwe_utils` defaulted to `{}`
    and `niah_utils` forwarded every model arg, so the `functools.cache`
    wrapper around `get_tokenizer` raised `TypeError: unhashable type: 'dict'`
    and the assertion that explains the problem was unreachable.
    """

    @pytest.mark.parametrize("builder", BUILDERS)
    def test_api_style_args_without_a_tokenizer_assert(self, builder):
        # What `--model_args model=...,base_url=...` parses into for an API
        # backend: no `tokenizer` and no `pretrained` key at all.
        with pytest.raises(AssertionError, match="No tokenizer or pretrained"):
            builder(base_url="http://localhost:8080/v1", num_concurrent=1)

    @pytest.mark.parametrize("builder", BUILDERS)
    def test_no_args_at_all_assert(self, builder):
        with pytest.raises(AssertionError, match="No tokenizer or pretrained"):
            builder()

    @pytest.mark.parametrize("builder", BUILDERS)
    def test_unhashable_model_args_do_not_leak_into_the_cache(self, builder):
        # A list-valued model arg used to reach functools.cache through
        # niah_utils' `get_tokenizer(**kwargs)` and raise TypeError there.
        with pytest.raises(AssertionError, match="No tokenizer or pretrained"):
            builder(stop=["\n", "###"], extra={"a": 1})
