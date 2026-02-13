import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
from typing_extensions import TypeVar

from lm_eval.api.registry import register_metric
from lm_eval.utils import softmax, warning_once


if TYPE_CHECKING:
    from lm_eval._types import LLResults

eval_logger = logging.getLogger(__name__)

_R = TypeVar("_R")
_T = TypeVar("_T")

__all__ = [
    "CorpusMetric",
    "AccAll",
    "Bleu",
    "Chrf",
    "Ter",
    "F1",
    "MCC",
    "BrierScore",
    "WordPerplexity",
    "BytePerplexity",
    "BitsPerByte",
]


class CorpusMetric(Generic[_R, _T], ABC):
    """Base class for corpus-level metrics.

    Corpus-level metrics are computed across multiple samples
    and typically require aggregation of intermediate results.

    Data flow::

        __call__(targets, results: _R) -> _T      # per document intermediate result
        aggregation(list[_T])          -> float   # corpus level
    """

    @abstractmethod
    def __call__(self, targets: Any, results: _R) -> _T:
        """Compute the per-item metric value for a single document."""
        ...

    @abstractmethod
    def aggregation(self, items: list[_T]) -> float:
        """Aggregate per-item values into a single corpus-level score."""
        ...

    def reduce(self, targets: list, results: list[_T], **kwargs) -> _T:
        """Collapse multiple repeats of a sample into one value. Corpus metrics only support repeat=1."""
        if len(results) != 1:
            warning_once(
                eval_logger,
                f"CorpusMetric {self.__class__.__name__} received multiple results; expected only one. Returning the first result.",
            )
        return results[0]


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
)
class BrierScore(CorpusMetric["LLResults", tuple[int, np.ndarray]]):
    """Brier score for multiple choice tasks.

    Computes the mean squared error between predicted probabilities
    (from softmax of log-likelihoods) and one-hot encoded targets.

    Lower scores are better (perfect score = 0.0).
    """

    def __call__(self, targets: Any, results: "LLResults") -> tuple[int, np.ndarray]:
        return results.target, softmax(np.array(results.lls))

    def aggregation(self, items: list[tuple[int, np.ndarray]]) -> float:
        gold, predictions = list(zip(*items, strict=True))
        bs, num_class = np.array(predictions).shape

        gold = list(gold)
        gold_one_hot = np.eye(num_class)[gold]
        return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


# ---------------------------------------------------------------------------
# Rolling loglikelihood corpus metrics
# ---------------------------------------------------------------------------


def _weighted_mean(items):
    a, b = zip(*items, strict=True)
    return sum(a) / sum(b)


@register_metric(
    metric="word_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
)
class WordPerplexity(CorpusMetric["LLResults", tuple[float, int]]):
    """Corpus-level word perplexity for rolling loglikelihood tasks.

    Computes the exponentiated average negative log-likelihood per word
    across all documents, weighted by word count.

    Lower scores are better.
    """

    def __call__(self, targets: Any, results: "LLResults") -> tuple[float, int]:
        return float(results.lls[results.target]), int(results.word_len[results.target])

    def aggregation(self, items: list[tuple[float, int]]) -> float:
        return math.exp(-_weighted_mean(items))


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
)
class BytePerplexity(CorpusMetric["LLResults", tuple[float, int]]):
    """Corpus-level byte perplexity for rolling loglikelihood tasks.

    Computes the exponentiated average negative log-likelihood per byte
    across all documents, weighted by byte count.

    Lower scores are better.
    """

    def __call__(self, targets: Any, results: "LLResults") -> tuple[float, int]:
        return float(results.lls[results.target]), int(results.byte_len[results.target])

    def aggregation(self, items: list[tuple[float, int]]) -> float:
        return math.exp(-_weighted_mean(items))


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
)
class BitsPerByte(CorpusMetric["LLResults", tuple[float, int]]):
    """Corpus-level bits-per-byte for rolling loglikelihood tasks.

    Converts the average negative log-likelihood per byte into bits
    by dividing by log(2), weighted by byte count across all documents.

    Lower scores are better.
    """

    def __call__(self, targets: Any, results: "LLResults") -> tuple[float, int]:
        return float(results.lls[results.target]), int(results.byte_len[results.target])

    def aggregation(self, items: list[tuple[float, int]]) -> float:
        return -_weighted_mean(items) / math.log(2)


# ---------------------------------------------------------------------------
# Sacrebleu helpers
# ---------------------------------------------------------------------------


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs, strict=True))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


# ---------------------------------------------------------------------------
# generate_until metrics
# ---------------------------------------------------------------------------


@register_metric(metric="bleu", higher_is_better=True, output_type="generate_until")
class Bleu(CorpusMetric[Any, tuple]):
    """BLEU score for generated text.

    The Bilingual Evaluation Understudy Score counts matching n-grams in the
    candidate translation to n-grams in the reference text.

    Higher is better.
    """

    import sacrebleu

    def __call__(self, targets: Any, results: Any) -> tuple:
        return targets, results

    def aggregation(self, items: list[tuple]) -> float:
        import sacrebleu

        refs, preds = zip(*items, strict=True)
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_bleu(preds, refs).score


@register_metric(metric="chrf", higher_is_better=True, output_type="generate_until")
class Chrf(CorpusMetric[Any, tuple]):
    """chrF++ score for generated text.

    chrF++ is based on character n-gram precision and recall
    enhanced with word n-grams.

    Higher is better.
    """

    import sacrebleu

    def __call__(self, targets: Any, results: Any) -> tuple:
        return targets, results

    def aggregation(self, items: list[tuple]) -> float:
        import sacrebleu

        refs, preds = zip(*items, strict=True)
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_chrf(preds, refs).score


@register_metric(metric="ter", higher_is_better=False, output_type="generate_until")
class Ter(CorpusMetric[Any, tuple]):
    """Translation Error Rate for generated text.

    Measures the number of edits required to change a system output
    into one of the references.

    Lower is better.
    """

    def __call__(self, targets: Any, results: Any) -> tuple:
        return targets, results

    def aggregation(self, items: list[tuple]) -> float:
        import sacrebleu

        refs, preds = zip(*items, strict=True)
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_ter(preds, refs).score


# ---------------------------------------------------------------------------
# multiple_choice corpus metrics
# ---------------------------------------------------------------------------


@register_metric(metric="f1", higher_is_better=True, output_type="multiple_choice")
class F1(CorpusMetric["LLResults", tuple[int, int]]):
    """F1 score for multiple choice tasks.

    Computes the maximum F1 score between gold labels and predicted labels
    (argmax of log-likelihoods).

    Higher is better.
    """

    def __call__(self, targets: Any, results: "LLResults") -> tuple[int, int]:
        pred = int(np.argmax(results.lls))
        return results.target, pred

    def aggregation(self, items: list[tuple[int, int]]) -> float:
        from sklearn.metrics import f1_score

        golds, preds = zip(*items, strict=True)
        return float(np.max(f1_score(golds, preds)))


@register_metric(metric="mcc", higher_is_better=True, output_type="multiple_choice")
class MCC(CorpusMetric["LLResults", tuple[int, int]]):
    """Matthews Correlation Coefficient for multiple choice tasks.

    Computes MCC between gold labels and predicted labels
    (argmax of log-likelihoods).

    Higher is better.
    """

    def __call__(self, targets: Any, results: "LLResults") -> tuple[int, int]:
        pred = int(np.argmax(results.lls))
        return results.target, pred

    def aggregation(self, items: list[tuple[int, int]]) -> float:
        from sklearn.metrics import matthews_corrcoef

        golds, preds = zip(*items, strict=True)
        return float(matthews_corrcoef(golds, preds))


@register_metric(
    metric="likelihood",
    higher_is_better=True,
    output_type="multiple_choice",
)
class Likelihood(CorpusMetric["LLResults", tuple[int, tuple]]):
    """Raw log-likelihoods of all choices paired with the gold index.

    Returns (gold_index, (ll_0, ll_1, ...)) for corpus-level custom aggregation.
    """

    def __call__(self, targets: Any, results: "LLResults") -> tuple[int, tuple]:
        return results.target, tuple(results.lls)

    def aggregation(self, items: list[tuple[int, tuple]]) -> float:
        from lm_eval.api.metrics import mean

        return mean([float(lls[gold]) for gold, lls in items])


# ---------------------------------------------------------------------------
# Loglikelihood: acc_all (BoolQ-style all-correct-per-question)
# ---------------------------------------------------------------------------


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
)
class AccAll(CorpusMetric["LLResults", tuple[int, dict]]):
    """All-correct accuracy for grouped questions (e.g. BoolQ).

    A question is scored as correct only if *every* answer option
    for that question is labeled correctly.
    """

    def __call__(self, targets: Any, results: "LLResults") -> tuple[int, dict]:
        pred = int(np.argmax(results.lls))
        gold = results.target
        return int(pred == gold), results.doc

    def aggregation(self, items: list[tuple[int, dict]]) -> float:
        question_scoring_dict: dict[tuple, list[bool]] = {}
        for pred, doc in items:
            key = (doc["idx"]["paragraph"], doc["idx"]["question"])
            if key not in question_scoring_dict:
                question_scoring_dict[key] = []
            gold_label = doc["label"] == 1
            question_scoring_dict[key].append(gold_label == pred)
        return float(np.mean([int(all(x)) for x in question_scoring_dict.values()]))
