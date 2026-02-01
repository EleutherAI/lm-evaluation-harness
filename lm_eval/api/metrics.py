import logging
import math
import os
import random
import re
import string
from collections.abc import Callable, Iterable, Sequence
from typing import Generic, TypeVar, cast

import numpy as np

from lm_eval import utils
from lm_eval.api.registry import register_aggregation, register_metric
from lm_eval.types import MCResult


T = TypeVar("T")
InputT = TypeVar("InputT")
IntermediateT = TypeVar("IntermediateT")

eval_logger = logging.getLogger(__name__)

# Conversion factor from natural log (nats) to bits
NAT_TO_BIT = 1.0 / math.log(2)


# Base Protocol for Metrics with __call__/Aggregate Pattern
class MetricProtocol(Generic[InputT, IntermediateT]):
    """Protocol for metrics with custom compute and aggregation stages.

    This pattern is useful for metrics that:
    1. Transform individual results (__call__ stage)
    2. Aggregate transformed results into a final score (aggregate stage)

    We especially use this for metrics that require corpus-level computation.

    Use __init__ to set up any state variables or kwargs
    """

    def __call__(self, input: InputT) -> IntermediateT:
        """Transform single result into intermediate representation."""
        raise NotImplementedError

    def aggregate(self, items: Iterable[IntermediateT]) -> float:
        """Aggregate intermediate results into final score."""
        raise NotImplementedError


# Register Aggregations First
@register_aggregation("bypass")
def bypass_agg(arr):
    return 999


@register_aggregation("nanmean")
def nanmean(arr):
    if len(arr) == 0 or all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@register_aggregation("median")
def median(arr):
    return arr[len(arr) // 2]


# Certain metrics must be calculated across all documents in a benchmark.
# We use them as aggregation metrics, paired with no-op passthrough metric fns.
@register_aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
)
class BrierScore:
    """Brier score for multiple choice tasks.

    Computes the mean squared error between predicted probabilities
    (from softmax of log-likelihoods) and one-hot encoded targets.

    Lower scores are better (perfect score = 0.0).
    """

    def __call__(self, result: MCResult):
        """Extract target and compute softmax probabilities from log-likelihoods.

        Args:
            result: MCResult containing target index and log-likelihoods

        Returns:
            Tuple of (target_index, probability_distribution)
        """
        return cast("int", result.target), utils.softmax(result.lls)

    def aggregate(self, items) -> float:
        """Compute mean Brier score across all items.

        Args:
            items: Iterable of (target, probabilities) tuples

        Returns:
            Mean Brier score (mean squared error)
        """
        gold, predictions = list(zip(*items, strict=True))
        bs, num_class = np.array(predictions).shape

        gold = list(gold)
        gold_one_hot = np.eye(num_class)[gold]
        return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(result: MCResult):
    return 1.0 if np.argmax(result.lls) == result.target else 0.0


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(result: MCResult):
    return 1.0 if np.argmax(result.lls_mutual_info) == result.target else 0.0


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(result: MCResult):
    return (
        1.0
        if np.argmax(np.array(result.lls) / np.array(result.char_lens)) == result.target
        else 0.0
    )


@register_metric(
    metric="acc_bytes",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_bytes_fn(result: MCResult):
    return (
        1.0
        if np.argmax(np.array(result.lls) / np.array(result.byte_lens)) == result.target
        else 0.0
    )


@register_metric(
    metric="bpb",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="mean",
)
def bpb_fn(result: MCResult) -> float:
    """Bits per byte for the correct choice.

    Measures information content normalized by byte length.
    Lower is better.
    """
    log_probs = np.array(result.lls)
    byte_lengths = np.array(result.byte_lens)
    target = result.target
    correct_logprob = log_probs[target]
    return float((-correct_logprob / byte_lengths[target]) * NAT_TO_BIT)


@register_metric(
    metric="logprob",
    higher_is_better=True,
    output_type=["multiple_choice"],
    aggregation="mean",
)
def logprob_fn(result: MCResult) -> float:
    """Log probability of the correct choice.

    Higher (less negative) is better.
    """
    log_probs = np.array(result.lls)
    target = result.target
    return float(log_probs[target])


@register_metric(
    metric="choice_logprob",
    higher_is_better=True,
    output_type=["multiple_choice"],
    aggregation="mean",
)
def choice_logprob_fn(result: MCResult) -> float:
    """Normalized log probability of the correct choice.

    Log probability after softmax normalization (log of the probability mass
    assigned to the correct choice). Higher is better.
    """
    log_probs = np.array(result.lls)
    normalized_log_probs = log_probs - np.logaddexp.reduce(log_probs)
    target = result.target
    return float(normalized_log_probs[target])


@register_metric(
    metric="choice_prob_norm",
    higher_is_better=True,
    output_type=["multiple_choice"],
    aggregation="mean",
)
def choice_prob_norm_fn(result: MCResult) -> float:
    """BPB-weighted probability of the correct choice.

    Probability of the correct choice when weighting by bits-per-byte
    (lower BPB gets higher weight). Higher is better.
    """
    log_probs = np.array(result.lls)
    byte_lengths = np.array(result.byte_lens)
    bpb_values = (-log_probs / byte_lengths) * NAT_TO_BIT
    bpb_weights = np.exp(-bpb_values)
    bpb_weights /= max(bpb_weights.sum(), 1e-8)  # avoid division by zero
    target = result.target
    return float(bpb_weights[target])


@register_metric(
    metric="choice_logprob_norm",
    higher_is_better=True,
    output_type=["multiple_choice"],
    aggregation="mean",
)
def choice_logprob_norm_fn(result: MCResult) -> float:
    """Log of BPB-weighted probability of the correct choice.

    Log probability of the correct choice when weighting by bits-per-byte.
    Higher is better.
    """
    log_probs = np.array(result.lls)
    byte_lengths = np.array(result.byte_lens)
    bpb_values = (-log_probs / byte_lengths) * NAT_TO_BIT
    bpb_weights = np.exp(-bpb_values)
    bpb_weights /= max(bpb_weights.sum(), 1e-8)  # avoid division by zero
    target = result.target
    correct_choice_prob_norm = float(bpb_weights[target])
    return float(np.log(correct_choice_prob_norm + 1e-30))


### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


###


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
)
class Perplexity:
    """Perplexity metric for language modeling.

    Computes exp(-mean(log_likelihoods)) to measure how well
    a probability model predicts a sample.

    Lower is better (perfect score = 1.0).
    """

    def __call__(self, items: float) -> float:
        """Pass through log likelihood values."""
        return items

    def aggregate(self, items: list[float]) -> float:
        """Compute perplexity from mean negative log likelihood."""
        return math.exp(-mean(items))


@register_metric(
    metric="likelihood",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def likelihood_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="word_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
)
class WordPerplexity:
    """Word-level perplexity for language modeling.

    Similar to Perplexity but uses weighted averaging for rolling contexts.
    """

    def __call__(self, items: tuple[float, float]) -> tuple[float, float]:
        """Pass through (log_likelihood, weight) tuples."""
        return items

    def aggregate(self, items: list[tuple[float, float]]) -> float:
        """Compute word perplexity using weighted mean."""
        return math.exp(-weighted_mean(items))


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
)
class BytePerplexity:
    """Byte-level perplexity for language modeling.

    Similar to WordPerplexity but measured at the byte level.
    """

    def __call__(self, items: tuple[float, float]) -> tuple[float, float]:
        """Pass through (log_likelihood, weight) tuples."""
        return items

    def aggregate(self, items: list[tuple[float, float]]) -> float:
        """Compute byte perplexity using weighted mean."""
        return math.exp(-weighted_mean(items))


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
)
class BitsPerByte:
    """Bits per byte metric for compression and language modeling.

    Measures the average number of bits needed to encode each byte.
    Lower is better.
    """

    def __call__(self, items: tuple[float, float]) -> tuple[float, float]:
        """Pass through (log_likelihood, weight) tuples."""
        return items

    def aggregate(self, items: list[tuple[float, float]]) -> float:
        """Compute bits per byte from weighted mean."""
        return -weighted_mean(items) / math.log(2)


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr: Sequence[T]) -> float:
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


@register_metric(
    metric="bypass",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice", "generate_until"],
    aggregation="bypass",
)
def bypass(items):
    return None


@register_metric(
    metric="mcc",
    higher_is_better=True,
    output_type="multiple_choice",
)
class MatthewsCorrelationCoef:
    """Matthews Correlation Coefficient for classification tasks.

    MCC is a balanced measure that can be used even if classes are
    of very different sizes. Returns a value between -1 and +1.
    """

    def __call__(
        self, items: tuple[int | str, int | str]
    ) -> tuple[int | str, int | str]:
        """Pass through (gold, prediction) pairs."""
        return items

    def aggregate(self, items: Iterable[tuple[int | str, int | str]]) -> float:
        """Compute Matthews Correlation Coefficient from all predictions."""
        from sklearn.metrics import matthews_corrcoef

        unzipped_list = list(zip(*items, strict=True))
        golds = unzipped_list[0]
        preds = unzipped_list[1]
        return matthews_corrcoef(golds, preds)


@register_metric(
    metric="f1",
    higher_is_better=True,
    output_type="multiple_choice",
)
class F1Score:
    """F1 score for classification tasks.

    Harmonic mean of precision and recall.
    Higher is better (perfect score = 1.0).
    """

    def __call__(
        self, items: tuple[int | str, int | str]
    ) -> tuple[int | str, int | str]:
        """Pass through (gold, prediction) pairs."""
        return items

    def aggregate(self, items: Iterable[tuple[int | str, int | str]]) -> float:
        """Compute F1 score from all predictions."""
        from sklearn.metrics import f1_score

        unzipped_list = list(zip(*items, strict=True))
        golds = unzipped_list[0]
        preds = unzipped_list[1]
        fscore = f1_score(golds, preds)
        return np.max(fscore)


@register_metric(
    metric="bleu",
    higher_is_better=True,
    output_type="generate_until",
)
class BLEU:
    """BLEU score for machine translation evaluation.

    Bilingual Evaluation Understudy Score - measures n-gram overlap
    between generated and reference translations.

    Higher is better (0-100 scale).
    """

    def __call__(self, items: tuple[str, str]) -> tuple[str, str]:
        """Pass through (reference, prediction) pairs."""
        return items

    def aggregate(self, items: Iterable[tuple[str, str]]) -> float:
        """Compute corpus-level BLEU score."""
        import sacrebleu

        refs = list(zip(*items, strict=True))[0]
        preds = list(zip(*items, strict=True))[1]
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_bleu(preds, refs).score


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type="generate_until",
)
class ChrF:
    """ChrF++ score for machine translation evaluation.

    Character n-gram F-score enhanced with word n-grams.
    More robust to morphological variations than BLEU.

    Higher is better.
    """

    def __call__(self, items: tuple[str, str]) -> tuple[str, str]:
        """Pass through (reference, prediction) pairs."""
        return items

    def aggregate(self, items: Iterable[tuple[str, str]]) -> float:
        """Compute corpus-level ChrF score."""
        import sacrebleu

        refs = list(zip(*items, strict=True))[0]
        preds = list(zip(*items, strict=True))[1]
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_chrf(preds, refs).score


@register_metric(
    metric="ter",
    higher_is_better=False,
    output_type="generate_until",
)
class TER:
    """Translation Error Rate for machine translation.

    Measures the number of edits required to change a system output
    into a reference translation.

    Lower is better (0 = perfect match).
    """

    def __call__(self, items: tuple[str, str]) -> tuple[str, str]:
        """Pass through (reference, prediction) pairs."""
        return items

    def aggregate(self, items: Iterable[tuple[str, str]]) -> float:
        """Compute corpus-level TER score."""
        import sacrebleu

        refs = list(zip(*items, strict=True))[0]
        preds = list(zip(*items, strict=True))[1]
        refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_ter(preds, refs).score


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items, strict=True))[0]
    docs = list(zip(*items, strict=True))[1]

    for doc, pred in zip(docs, preds, strict=True):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items, strict=True))[0]
    docs = list(zip(*items, strict=True))[1]

    for doc, pred in zip(docs, preds, strict=True):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def weighted_mean(items):
    a, b = zip(*items, strict=True)
    return sum(a) / sum(b)


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


# stderr stuff


class _bootstrap_internal:
    """
    Pool worker: `(i, xs)` → `n` bootstrap replicates
    of `f(xs)`using a RNG seeded with `i`.
    """

    def __init__(self, f: Callable[[Sequence[T]], float], n: int) -> None:
        self.f = f
        self.n = n

    def __call__(self, v: tuple[int, Sequence[T]]) -> list[float]:
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def _bootstrap_internal_no_mp(
    f: Callable[[Sequence[T]], float], xs: Sequence[T], iters: int
) -> list[float]:
    """
    Single-process fallback: compute `iters` bootstrap replicates
    of statistic`f(xs)`, chunked (≤ 1000 draws).
    """
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print(f"bootstrapping for stddev: {f.__name__}")

    # A single loop replaces the multiprocessing pool.
    for i in tqdm(range(iters // chunk_size)):
        rnd = random.Random(i)
        for _ in range(chunk_size):
            res.append(f(rnd.choices(xs, k=len(xs))))

    return res


def bootstrap_stderr(
    f: Callable[[Sequence[T]], float], xs: Sequence[T], iters: int
) -> float:
    """
    Bootstrap estimate of the standard error of statistic `f(xs)`
    using up to `iters` resamples, chunked (≤ 1000 draws)

    Executes in parallel unless the env-var `DISABLE_MULTIPROC` is set;
    """
    if not os.getenv("DISABLE_MULTIPROC"):
        import multiprocessing as mp

        # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
        # equivalent to stderr calculated without Bessel's correction in the stddev.
        # Unfortunately, I haven't been able to figure out what the right correction is
        # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
        # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
        # Thankfully, shouldn't matter because our samples are pretty big usually anyways
        res = []
        chunk_size = min(1000, iters)
        from tqdm import tqdm

        print("bootstrapping for stddev:", f.__name__)
        with mp.Pool(mp.cpu_count()) as pool:
            for bootstrap in tqdm(
                pool.imap(
                    _bootstrap_internal(f, chunk_size),
                    [(i, xs) for i in range(iters // chunk_size)],
                ),
                total=iters // chunk_size,
            ):
                # sample w replacement
                res.extend(bootstrap)
    else:
        res = _bootstrap_internal_no_mp(f, xs, iters)

    return sample_stddev(res)


def stderr_for_metric(
    metric: Callable[[Sequence[T]], float], bootstrap_iters: int
) -> Callable[[Sequence[T]], float] | None:
    """
    Return a function that estimates the standard error of `metric(xs)`.

    * If `bootstrap_iters > 0` and the metric is in the pre-approved
      bootstrappable list, use `bootstrap_stderr` with that many draws.
    * If the metric has a closed-form SE (e.g. `mean`, `acc_all`), use it.
    * Otherwise, return `None`.
    """

    if bootstrap_iters <= 0:
        # return no function (don't compute stderr) if bootstrap iters = 0
        return None

    # TODO: bootstrappable list should be able to set class based metrics too
    bootstrappable = [
        median,
        # matthews_corrcoef,
        # f1_score,
        perplexity,
        # bleu,
        # chrf,
        # ter,
        nanmean,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric)


def pooled_sample_stderr(stderrs: list[float], sizes: list[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum(
            [
                (size - 1) * stderr**2 * size
                for size, stderr in zip(sizes, stderrs, strict=True)
            ]
        )
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def combined_sample_stderr(stderrs: list[float], sizes: list[int], metrics=None):
    assert metrics is not None, (
        "Need to pass a list of each subtask's metric for this stderr aggregation"
    )
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)

    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1390 for more documentation.
    # This formula depends on sample means.
    # removed because it seems to give erroneously huge stderrs for groupings of tasks
    # and does not seem to match up with bootstrap-calculated stderrs for groups.

    ### don't use this unless a statistician has told you it's the right thing to do ###

    # accumulators: we'll aggregate pairwise N - 1 times
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:], strict=True):
        curr_score = ((curr_score * curr_size) + (score * size)) / (
            curr_size + size
        )  # NOTE: this assumes our aggregation fn is "mean"

        variance = ((curr_size - 1) * variance + (size - 1) * (stderr**2)) / (
            curr_size + size - 1
        ) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (
            curr_score - score
        ) ** 2

    return np.sqrt(variance)


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    # TODO: does not hold for non-mean aggregations
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum(
        [metric * size for metric, size in zip(metrics, sizes, strict=True)]
    ) / sum(sizes)
