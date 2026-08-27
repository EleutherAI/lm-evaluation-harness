"""Regression tests for the RULER synthetic samplers.

Every sampler sizes its prompts by shrinking the context until it fits
`max_seq_length`. When the smallest context a sample can have still overflows,
the shrinking loops used to stop making progress and spin forever instead of
giving up on the sample.
"""

import pytest

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
