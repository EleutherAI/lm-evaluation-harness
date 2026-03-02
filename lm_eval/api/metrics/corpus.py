from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic
from typing_extensions import TypeVar

import numpy as np

from lm_eval.api.registry import register_metric as metric
from lm_eval.utils import warning_once


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import float64
    from numpy._typing import NDArray

    from .results import LLResults

    GenPred = list[str]
    LLPred = LLResults

eval_logger = logging.getLogger(__name__)

_R = TypeVar("_R", bound="LLResults | GenPred")
_T = TypeVar("_T")

__all__ = [
    "F1",
    "MCC",
    "BitsPerByte",
    "Bleu",
    "BytePerplexity",
    "Chrf",
    "CorpusMetric",
    "Perplexity",
    "Ter",
    "WordPerplexity",
]


class CorpusMetric(ABC, Generic[_R, _T]):
    """Base class for corpus-level metrics.

    Corpus-level metrics are computed across multiple samples
    and typically require aggregation of intermediate results.

    Data flow:
        ``__call__(references, predictions: _R) -> _T`` # per document intermediate result
        ``aggregation(list[_T]) -> float`` # corpus level
    """

    @abstractmethod
    def __call__(self, references: Any, predictions: _R) -> _T:
        """Compute the per-item metric value for a single document."""
        ...

    @abstractmethod
    def aggregation(self, items: Sequence[_T]) -> float:
        """Aggregate per-item values into a single corpus-level score."""
        ...

    def reduce(
        self, references: Sequence[Any], predictions: Sequence[_T], **kwargs
    ) -> _T:
        """Collapse multiple repeats of a sample into one value. Corpus metrics only support repeat=1."""
        if len(predictions) != 1:
            warning_once(
                eval_logger,
                f"CorpusMetric {self.__class__.__name__} received multiple results; expected only one. Returning the first result.",
            )
        return predictions[0]


class _BrierScore(CorpusMetric["LLResults", float]):
    """Brier score for multiple choice tasks.

    We use the functional form. Only here for simplicity.
    """

    def __call__(self, references: Any, predictions: LLResults) -> float:
        from .ll import brier_score

        return brier_score(references, predictions)

    def aggregation(self, items: Sequence[float]) -> float:
        return sum(items) / len(items)


# ---------------------------------------------------------------------------
# Loglikelihood perplexity
# ---------------------------------------------------------------------------


@metric("perplexity", higher_is_better=False, output_type="loglikelihood")
class Perplexity(CorpusMetric["LLResults", float]):
    """Corpus-level perplexity for loglikelihood tasks.

    Per-document: extracts the gold log-likelihood.
    Aggregation: ``exp(-mean(lls))`` across all documents.
    """

    def __call__(self, references: int | list[int], predictions: LLResults) -> float:
        if len(predictions.lls) == 1:
            return float(predictions.lls[0])
        return float(predictions.lls[references])

    def aggregation(self, items: Sequence[float]) -> float:
        return math.exp(-sum(items) / len(items))


# ---------------------------------------------------------------------------
# Rolling loglikelihood corpus metrics
# ---------------------------------------------------------------------------


def _weighted_mean(items):
    a, b = zip(*items, strict=True)
    return sum(a) / sum(b)


@metric("word_perplexity", higher_is_better=False, output_type="loglikelihood_rolling")
class WordPerplexity(CorpusMetric["LLResults", tuple[float, int]]):
    """Corpus-level word perplexity for rolling loglikelihood tasks.

    Computes the exponentiated average negative log-likelihood per word
    across all documents, weighted by word count.

    Lower scores are better.
    """

    def __call__(self, references: int, predictions: LLResults) -> tuple[float, int]:
        return float(predictions.lls[references]), int(
            predictions.word_len()[references]
        )

    def aggregation(self, items: Sequence[tuple[float, int]]) -> float:
        return math.exp(-_weighted_mean(items))


@metric(
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

    def __call__(self, references: int, predictions: LLResults) -> tuple[float, int]:
        return float(predictions.lls[references]), int(
            predictions.byte_len()[references]
        )

    def aggregation(self, items: Sequence[tuple[float, int]]) -> float:
        return math.exp(-_weighted_mean(items))


@metric(
    metric="bits_per_byte", higher_is_better=False, output_type="loglikelihood_rolling"
)
class BitsPerByte(CorpusMetric["LLResults", tuple[float, int]]):
    """Corpus-level bits-per-byte for rolling loglikelihood tasks.

    Converts the average negative log-likelihood per byte into bits
    by dividing by log(2), weighted by byte count across all documents.

    Lower scores are better.
    """

    def __call__(self, references: int, predictions: LLResults) -> tuple[float, int]:
        return float(predictions.lls[references]), int(
            predictions.byte_len()[references]
        )

    def aggregation(self, items: Sequence[tuple[float, int]]) -> float:
        return -_weighted_mean(items) / math.log(2)


# ---------------------------------------------------------------------------
# Sacrebleu helpers
# ---------------------------------------------------------------------------


def _is_non_str_iterable(obj: object) -> bool:
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(
    refs: Sequence[str] | Sequence[Sequence[str]],
    preds: Sequence[str] | Sequence[Sequence[str]],
) -> tuple[list[tuple[str, ...]], list[str]]:
    """Format refs and preds for sacrebleu corpus calculation. It is very particular."""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not _is_non_str_iterable(refs):
        refs = list(refs)
    if not _is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]  # type:ignore[invalid-assignment]
    refs = list(zip(*refs, strict=True))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not _is_non_str_iterable(preds):
        preds = list(preds)
    if _is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds  # type:ignore[invalid-return-type]


# ---------------------------------------------------------------------------
# generate_until metrics
# ---------------------------------------------------------------------------


class _SacrebleuCorpusMetric(CorpusMetric["GenPred", tuple[list[str], list[str]]]):
    """Base for sacrebleu corpus metrics (BLEU, chrF, TER).

    With repeat > 1, predictions is ``list[str]`` with one string per repeat
    while references stays the same.  ``reduce`` keeps only the first repeat
    prediction so that downstream ``_sacreformat`` / ``aggregation`` see
    exactly one prediction per sample.
    """

    def __call__(
        self, references: list[str], predictions: list[str]
    ) -> tuple[list[str], list[str]]:
        return references, predictions

    def reduce(
        self,
        references: Sequence[Any],
        predictions: Sequence[tuple[list[str], list[str]]],
        **kwargs,
    ) -> tuple[list[str], list[str]]:
        refs, preds = predictions[0]
        if len(preds) > 1:
            warning_once(
                eval_logger,
                f"CorpusMetric {self.__class__.__name__} received {len(preds)} "
                "repeat predictions; keeping only the first.",
            )
            preds = preds[:1]
        return refs, preds


@metric("bleu", higher_is_better=True, output_type="generate_until")
class Bleu(_SacrebleuCorpusMetric):
    """BLEU score for generated text.

    The Bilingual Evaluation Understudy Score counts matching n-grams in the
    candidate translation to n-grams in the reference text.

    Higher is better.
    """

    def aggregation(self, items: Sequence[tuple[list[str], list[str]]]) -> float:
        import sacrebleu

        refs, preds = zip(*items, strict=True)
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_bleu(preds, refs).score


@metric("chrf", higher_is_better=True, output_type="generate_until")
class Chrf(_SacrebleuCorpusMetric):
    """chrF++ score for generated text.

    chrF++ is based on character n-gram precision and recall
    enhanced with word n-grams.

    Higher is better.
    """

    def aggregation(self, items: Sequence[tuple[list[str], list[str]]]) -> float:
        import sacrebleu

        refs, preds = zip(*items, strict=True)
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_chrf(preds, refs).score


@metric("ter", higher_is_better=False, output_type="generate_until")
class Ter(_SacrebleuCorpusMetric):
    """Translation Error Rate for generated text.

    Measures the number of edits required to change a system output
    into one of the references.

    Lower is better.
    """

    def aggregation(self, items: Sequence[tuple[list[str], list[str]]]) -> float:
        import sacrebleu

        refs, preds = zip(*items, strict=True)
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_ter(preds, refs).score


# ---------------------------------------------------------------------------
# multiple_choice corpus metrics
# ---------------------------------------------------------------------------


@metric("f1", higher_is_better=True, output_type="multiple_choice")
class F1(CorpusMetric["LLResults", tuple[int, int]]):
    """F1 score for multiple choice tasks.

    Computes the maximum F1 score between gold labels and predicted labels
    (argmax of log-likelihoods).

    Higher is better.
    """

    def __call__(self, references: Any, predictions: LLResults) -> tuple[int, int]:
        pred = int(np.argmax(predictions.lls))
        return references, pred

    def aggregation(self, items: Sequence[tuple[int, int]]) -> float:
        from sklearn.metrics import f1_score

        golds, preds = zip(*items, strict=True)
        return float(np.max(f1_score(golds, preds)))


@metric("mcc", higher_is_better=True, output_type="multiple_choice")
class MCC(CorpusMetric["LLResults", tuple[int, int]]):
    """Matthews Correlation Coefficient for multiple choice tasks.

    Computes MCC between gold labels and predicted labels
    (argmax of log-likelihoods).

    Higher is better.
    """

    def __call__(self, references: Any, predictions: LLResults) -> tuple[int, int]:
        pred = int(np.argmax(predictions.lls))
        return references, pred

    def aggregation(self, items: Sequence[tuple[int, int]]) -> float:
        from sklearn.metrics import matthews_corrcoef

        golds, preds = zip(*items, strict=True)
        return float(matthews_corrcoef(golds, preds))


@metric("likelihood", higher_is_better=True, output_type="multiple_choice")
class Likelihood(CorpusMetric["LLResults", tuple[int, "tuple[NDArray[float64], ...]"]]):
    """Raw log-likelihoods of all choices paired with the gold index.

    Returns (gold_index, (ll_0, ll_1, ...)) for corpus-level custom aggregation.
    """

    def __call__(self, references: int, predictions: LLResults) -> tuple[int, tuple]:
        return references, tuple(predictions.lls)

    def aggregation(
        self, items: Sequence[tuple[int, tuple[NDArray[float64], ...]]]
    ) -> float:
        from .aggregations import mean

        return mean([float(lls[gold]) for gold, lls in items])


# # ---------------------------------------------------------------------------
# # Loglikelihood: acc_all (BoolQ-style all-correct-per-question)
# # ---------------------------------------------------------------------------
#
#
# @metric(
#     metric="acc_all",
#     higher_is_better=True,
#     output_type="loglikelihood",
# )
# class AccAll(CorpusMetric["LLResults", tuple[int, dict]]):
#     """All-correct accuracy for grouped questions (e.g. BoolQ).
#
#     A question is scored as correct only if *every* answer option
#     for that question is labeled correctly.
#     """
#
#     def __call__(self, references: int, predictions: "LLResults") -> tuple[int, dict]:
#         pred = int(np.argmax(predictions.lls))
#         gold = references
#         return int(pred == gold), predictions.doc
#
#     def aggregation(self, items: list[tuple[int, dict]]) -> float:
#         question_scoring_dict: dict[tuple, list[bool]] = {}
#         for pred, doc in items:
#             key = (doc["idx"]["paragraph"], doc["idx"]["question"])
#             if key not in question_scoring_dict:
#                 question_scoring_dict[key] = []
#             gold_label = doc["label"] == 1
#             question_scoring_dict[key].append(gold_label == pred)
#         return float(np.mean([int(all(x)) for x in question_scoring_dict.values()]))
