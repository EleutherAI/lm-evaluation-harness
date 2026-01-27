import logging
import math
import os
import random
import re
import string
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, TypeVar

import numpy as np
import sacrebleu

from lm_eval.api.registry import register_aggregation, register_metric

try:
    import openai
    OPENAI_AVAILABLE = True
    # Suppress verbose httpx logging from OpenAI client
    logging.getLogger("httpx").setLevel(logging.WARNING)
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    from genson import SchemaBuilder
    GENSON_AVAILABLE = True
except ImportError:
    GENSON_AVAILABLE = False

try:
    import tenacity
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

import json


T = TypeVar("T")

eval_logger = logging.getLogger(__name__)

# Module-level storage for deferred LLM judge details saving
# This allows the EvaluationTracker to save details with proper model paths
_llm_judge_pending_details: dict = {}


def get_pending_llm_judge_details() -> dict:
    """
    Retrieve and clear pending LLM judge details for saving.

    Returns a dict mapping task_name to details dict containing:
    - judge_model: The LLM judge model name
    - results: List of individual judgment results

    This function clears the pending details after retrieval.
    """
    global _llm_judge_pending_details
    details = _llm_judge_pending_details.copy()
    _llm_judge_pending_details.clear()
    return details


def _infer_json_schema_from_example(example: dict, name: str = "response") -> dict:
    """
    Infer a JSON schema from an example dict using genson.

    :param example: A dict with example values (e.g., {"score": 8.5, "explanation": "Good translation"})
    :param name: Name for the schema (used for structured outputs)
    :return: JSON schema dict compatible with OpenAI structured outputs
    """
    if not GENSON_AVAILABLE:
        raise ImportError(
            "genson package is required for JSON schema inference. Install with: pip install genson"
        )

    builder = SchemaBuilder()
    builder.add_object(example)
    schema = builder.to_schema()

    # OpenAI structured outputs requires additionalProperties: false for strict mode
    if "properties" in schema:
        schema["additionalProperties"] = False
        # Add required field listing all properties for strict mode
        schema["required"] = list(schema["properties"].keys())

    return schema


def _format_response_format_example(response_format: dict) -> str:
    """
    Format the response_format example as a human-readable JSON string for prompts.

    :param response_format: Dict with example values
    :return: Formatted JSON string for inclusion in prompts
    """
    return json.dumps(response_format, indent=2)


def _is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable (transient errors).

    Retryable errors include:
    - Rate limit errors (429)
    - Server errors (500, 502, 503, 504)
    - Connection/timeout errors

    Non-retryable errors include:
    - Authentication errors (401, 403)
    - Bad request errors (400)
    - Not found errors (404)
    """
    if not OPENAI_AVAILABLE:
        return False

    # OpenAI specific error handling
    if isinstance(exception, openai.RateLimitError):
        return True
    if isinstance(exception, openai.APIConnectionError):
        return True
    if isinstance(exception, openai.InternalServerError):
        return True
    if isinstance(exception, openai.APITimeoutError):
        return True

    # Non-retryable OpenAI errors
    if isinstance(exception, openai.AuthenticationError):
        return False
    if isinstance(exception, openai.BadRequestError):
        return False
    if isinstance(exception, openai.NotFoundError):
        return False
    if isinstance(exception, openai.PermissionDeniedError):
        return False

    # For generic exceptions, check status code if available
    if hasattr(exception, "status_code"):
        status = exception.status_code
        return status in (429, 500, 502, 503, 504)

    # Default: don't retry unknown exceptions
    return False


def _preflight_check_llm_judge(
    api_base: Optional[str],
    api_key: Optional[str],
    model: str,
    judge_label: str = "LLM Judge",
) -> None:
    """
    Perform a pre-flight check to verify API connectivity and authentication.

    Makes a minimal API call to verify:
    - API endpoint is reachable
    - API key is valid
    - Model name is correct

    :param api_base: Base URL for API
    :param api_key: API key
    :param model: Model name
    :param judge_label: Label for logging
    :raises: Exception if pre-flight check fails
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai package is required for LLM-as-a-Judge metrics. Install with: pip install openai"
        )

    eval_logger.info(f"  Running pre-flight API check...")

    client_kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
    if api_base:
        client_kwargs["base_url"] = api_base

    client = openai.OpenAI(**client_kwargs)

    try:
        # Minimal API call to verify connectivity
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=5,
            temperature=0.0,
        )
        eval_logger.info(f"  Pre-flight check: PASSED")
    except openai.AuthenticationError as e:
        raise RuntimeError(
            f"{judge_label} pre-flight check FAILED: Authentication error. "
            f"Check your API key. Error: {e}"
        ) from e
    except openai.NotFoundError as e:
        raise RuntimeError(
            f"{judge_label} pre-flight check FAILED: Model '{model}' not found. "
            f"Check the model name. Error: {e}"
        ) from e
    except openai.PermissionDeniedError as e:
        raise RuntimeError(
            f"{judge_label} pre-flight check FAILED: Permission denied. "
            f"Check your API key permissions. Error: {e}"
        ) from e
    except openai.APIConnectionError as e:
        raise RuntimeError(
            f"{judge_label} pre-flight check FAILED: Cannot connect to API at "
            f"'{api_base or 'https://api.openai.com/v1'}'. Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"{judge_label} pre-flight check FAILED: {type(e).__name__}: {e}"
        ) from e


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


@register_aggregation("f1")
def f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds)

    return np.max(fscore)


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    from sklearn.metrics import matthews_corrcoef

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


@register_aggregation("bleu")
def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_aggregation("chrf")
def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better  # TODO I think
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_aggregation("ter")
def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


@register_aggregation("brier_score")
def brier_score(items):  # This is a passthrough function
    gold, predictions = list(zip(*items))
    bs, num_class = np.array(predictions).shape

    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
    aggregation="brier_score",
)
def brier_score_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_bytes",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_bytes_fn(items):  # This is a passthrough function
    return items


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
    aggregation="perplexity",
)
def perplexity_fn(items):  # This is a passthrough function
    return items


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
    aggregation="weighted_perplexity",
)
def word_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="byte_perplexity",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="weighted_perplexity",
)
def byte_perplexity_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bits_per_byte",
    higher_is_better=False,
    output_type="loglikelihood_rolling",
    aggregation="bits_per_byte",
)
def bits_per_byte_fn(items):  # This is a passthrough function
    return items


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
    aggregation="matthews_corrcoef",
)
def mcc_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="f1",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="f1",
)
def f1_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="bleu",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="bleu",
)
def bleu_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="chrf",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="chrf",
)
def chrf_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="ter",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="ter",
)
def ter_fn(items):  # This is a passthrough function
    return items


@register_metric(
    metric="acc_all",
    higher_is_better=True,
    output_type="loglikelihood",
    aggregation="mean",
)
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
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
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
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
    a, b = zip(*items)
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
    refs = list(zip(*refs))
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
) -> Optional[Callable[[Sequence[T]], float]]:
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

    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        ter,
        nanmean,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)


def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def combined_sample_stderr(stderrs: List[float], sizes: List[int], metrics=None):
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

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:]):
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

    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)


# LLM-as-a-Judge metric implementation
# The metric uses a passthrough pattern (like BLEU) where the actual LLM calls
# happen in the aggregation function, allowing for batch concurrent processing.


def _render_llm_judge_prompt(
    prompt_template: str,
    prediction: str,
    reference: Optional[str] = None,
    doc: Optional[dict] = None,
    config: Optional[dict] = None,
) -> str:
    """
    Render a Jinja2 template for LLM judge prompts.

    Template has access to:
    - prediction: The model's generated response
    - reference: The gold/target answer
    - doc: The full document dict (fields also accessible directly)
    - response_format: Formatted JSON example string (if response_format is in config)
    - Any custom fields from the metric config (e.g., source_field, etc.)

    This allows YAML configs to define custom fields that can be used in templates:
        source_field: sentence_eng_Latn
        prompt_template: |
          Source: {{ doc[source_field] }}
          Translation: {{ prediction }}

    For structured outputs, if response_format is defined in config:
        response_format:
          score: 8.5
          explanation: "Good translation"
        prompt_template: |
          Evaluate this translation...
          {{ response_format }}

    If {{ response_format }} placeholder is not in the template but response_format
    is defined, instructions will be auto-appended to the prompt.
    """
    # Reserved keys that should not be passed to templates
    reserved_keys = frozenset({
        "prompt_template", "api_base", "api_key", "model",
        "temperature", "max_tokens", "concurrency",
        "save_details", "task_name", "metric", "aggregation",
        "higher_is_better", "hf_evaluate", "response_format",
        "score_field", "name"
    })

    # Check if response_format is defined in config
    response_format_dict = config.get("response_format") if config else None
    response_format_str = None
    if response_format_dict:
        response_format_str = _format_response_format_example(response_format_dict)

    if JINJA2_AVAILABLE:
        template = jinja2.Template(prompt_template)
        template_vars = {
            "prediction": prediction,
            "reference": reference,
            "doc": doc or {},
        }

        # Add formatted response_format example if defined
        if response_format_str:
            template_vars["response_format"] = response_format_str

        # Add custom config fields to template context
        # This allows templates to use {{ doc[source_field] }} patterns
        if config:
            for key, value in config.items():
                if key not in reserved_keys:
                    template_vars[key] = value

        # Also expose doc fields directly for convenience
        if doc:
            template_vars.update(doc)

        rendered = template.render(**template_vars)

        # If response_format is defined but {{ response_format }} was not in template,
        # auto-append JSON format instructions
        if response_format_str and "response_format" not in prompt_template:
            rendered += f"\n\nRespond with a JSON object in exactly this format:\n{response_format_str}"

        return rendered
    else:
        # Fallback to simple string formatting
        template_vars = {
            "prediction": prediction,
            "reference": reference if reference is not None else "N/A",
        }

        if response_format_str:
            template_vars["response_format"] = response_format_str

        if config:
            for key, value in config.items():
                if key not in reserved_keys:
                    template_vars[key] = value

        if doc:
            template_vars.update(doc)

        try:
            rendered = prompt_template.format(**template_vars)
        except KeyError as e:
            eval_logger.error(f"Error formatting prompt template: {e}")
            raise

        # If response_format is defined but not used in template, auto-append
        if response_format_str and "{response_format}" not in prompt_template:
            rendered += f"\n\nRespond with a JSON object in exactly this format:\n{response_format_str}"

        return rendered


def _call_llm_judge_single(
    prediction: str,
    reference: Optional[str] = None,
    doc: Optional[dict] = None,
    prompt_template: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    config: Optional[dict] = None,
    json_schema: Optional[dict] = None,
    score_field: str = "score",
    retry_attempts: int = 3,
    retry_min_wait: float = 1.0,
    retry_max_wait: float = 60.0,
) -> dict:
    """
    Make a single LLM judge API call with retry logic.

    :param prediction: The model's generated prediction/response
    :param reference: Optional reference/gold answer
    :param doc: Full document dict for Jinja template rendering
    :param prompt_template: Jinja template string
    :param api_base: Base URL for OpenAI-compatible API
    :param api_key: API key for authentication
    :param model: Model name to use for judging
    :param temperature: Sampling temperature
    :param max_tokens: Maximum tokens in response
    :param config: Full metric config dict for template variable access
    :param json_schema: Pre-computed JSON schema for structured outputs (optional)
    :param score_field: Field name to extract score from JSON response (default: "score")
    :param retry_attempts: Number of retry attempts for transient errors (default: 3)
    :param retry_min_wait: Minimum wait time between retries in seconds (default: 1.0)
    :param retry_max_wait: Maximum wait time between retries in seconds (default: 60.0)
    :return: Dict with score, judgment, formatted_prompt, and parsed fields
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai package is required for LLM-as-a-Judge metrics. Install with: pip install openai"
        )

    use_structured_outputs = json_schema is not None

    # Default prompt template
    if prompt_template is None:
        if use_structured_outputs:
            prompt_template = """You are an expert evaluator. Please evaluate the quality of the following response.
{% if doc %}
Context: {{ doc }}
{% endif %}
{% if reference %}
Reference Answer: {{ reference }}
{% endif %}
Generated Response: {{ prediction }}

Please provide a score from 0 to 10 and explain your reasoning.
{{ response_format }}"""
        else:
            prompt_template = """You are an expert evaluator. Please evaluate the quality of the following response.
{% if doc %}
Context: {{ doc }}
{% endif %}
{% if reference %}
Reference Answer: {{ reference }}
{% endif %}
Generated Response: {{ prediction }}

Please provide a score from 0 to 10 and explain your reasoning.
Your response should start with "Score: X.XX" on the first line."""

    # Render the prompt using Jinja
    formatted_prompt = _render_llm_judge_prompt(
        prompt_template=prompt_template,
        prediction=prediction,
        reference=reference,
        doc=doc,
        config=config,
    )

    # Set up API client
    client_kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
    if api_base:
        client_kwargs["base_url"] = api_base

    client = openai.OpenAI(**client_kwargs)

    def make_api_call():
        """Inner function that makes the actual API call (can be retried)."""
        if use_structured_outputs:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "llm_judge_response",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            )
        else:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

    # Apply retry logic if tenacity is available and retries are enabled
    if TENACITY_AVAILABLE and retry_attempts > 0:
        make_api_call = tenacity.retry(
            stop=tenacity.stop_after_attempt(retry_attempts),
            wait=tenacity.wait_exponential(min=retry_min_wait, max=retry_max_wait),
            retry=tenacity.retry_if_exception(_is_retryable_error),
            before_sleep=lambda retry_state: eval_logger.warning(
                f"LLM judge API call failed (attempt {retry_state.attempt_number}/{retry_attempts}), "
                f"retrying in {retry_state.next_action.sleep:.1f}s: {retry_state.outcome.exception()}"
            ),
            reraise=True,
        )(make_api_call)

    # Make API call (with or without retries)
    try:
        response = make_api_call()

        if use_structured_outputs:
            judgment_text = response.choices[0].message.content

            # Parse JSON response
            try:
                judgment_parsed = json.loads(judgment_text)
                score = float(judgment_parsed.get(score_field, 0.0))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                eval_logger.warning(f"Failed to parse JSON response: {e}. Raw: {judgment_text}")
                score = 0.0
                judgment_parsed = None

            return {
                "score": score,
                "judgment_raw": judgment_text,
                "judgment_parsed": judgment_parsed,
                "formatted_prompt": formatted_prompt,
                "prediction": prediction,
                "reference": reference,
                "error": None,
            }

        else:
            judgment_text = response.choices[0].message.content

            # Parse score - NO CLAMPING
            score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", judgment_text, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
            else:
                eval_logger.warning(
                    f"Could not parse score from LLM judge response. Defaulting to 0.0"
                )
                score = 0.0

            # Extract explanation (everything after the first line)
            lines = judgment_text.strip().split("\n")
            explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            return {
                "score": score,
                "judgment_raw": judgment_text,
                "explanation": explanation,
                "formatted_prompt": formatted_prompt,
                "prediction": prediction,
                "reference": reference,
                "error": None,
            }

    except Exception as e:
        eval_logger.error(f"Error calling LLM judge API (after retries): {e}")
        return {
            "score": 0.0,
            "judgment_raw": None,
            "explanation": None,
            "formatted_prompt": formatted_prompt,
            "prediction": prediction,
            "reference": reference,
            "error": str(e),
        }


@register_aggregation("llm_judge")
def llm_judge_agg(items):
    """
    Aggregates LLM judge evaluations by running concurrent API calls.

    Items are tuples of (reference, prediction, doc, config) collected from
    the passthrough metric function.

    Config dict can contain:
    - prompt_template: Jinja template string
    - api_base: API endpoint URL
    - api_key: API key (optional, defaults to OPENAI_API_KEY env var)
    - model: Model name (default: gpt-4)
    - temperature: Sampling temperature (default: 0.0)
    - max_tokens: Max response tokens (default: 1024)
    - concurrency: Number of concurrent API calls (default: 32)
    - save_details: Whether to save detailed JSONL results (default: true)
    - task_name: Name of the task (auto-populated from task config)

    When save_details is enabled, results are stored in _llm_judge_pending_details
    and can be retrieved via get_pending_llm_judge_details() for saving by
    EvaluationTracker, which ensures proper file paths with model_name_sanitized.

    :param items: List of (reference, prediction, doc, config) tuples
    :return: Mean score across all judgments
    """
    if not items:
        return np.nan

    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai package is required for LLM-as-a-Judge metrics. Install with: pip install openai"
        )

    # Extract config from first item
    first_item = items[0]
    if len(first_item) < 4:
        eval_logger.error(
            "llm_judge items must be (reference, prediction, doc, config) tuples"
        )
        return np.nan

    config = first_item[3]

    # Get optional name for this judge variant (e.g., "accuracy", "fluency")
    judge_name = config.get("name")

    # Build environment variable names based on judge name
    # E.g., name="accuracy" -> LLM_JUDGE_ACCURACY_API_BASE, LLM_JUDGE_ACCURACY_MODEL
    # Falls back to LLM_JUDGE_API_BASE, LLM_JUDGE_MODEL if name-specific not set
    if judge_name:
        name_upper = judge_name.upper()
        env_api_base = os.environ.get(f"LLM_JUDGE_{name_upper}_API_BASE") or os.environ.get("LLM_JUDGE_API_BASE")
        env_api_key = os.environ.get(f"LLM_JUDGE_{name_upper}_API_KEY") or os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        env_model = os.environ.get(f"LLM_JUDGE_{name_upper}_MODEL") or os.environ.get("LLM_JUDGE_MODEL")
    else:
        env_api_base = os.environ.get("LLM_JUDGE_API_BASE")
        env_api_key = os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        env_model = os.environ.get("LLM_JUDGE_MODEL")

    prompt_template = config.get("prompt_template")
    # YAML config takes precedence for api_base and model, then env vars, then defaults
    # api_key is ONLY from env vars to prevent accidental exposure in config files
    api_base = config.get("api_base") or env_api_base
    api_key = env_api_key  # Only from environment variables
    model = config.get("model") or env_model or "gpt-4"
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 1024))
    concurrency = int(config.get("concurrency", 32))
    save_details = config.get("save_details", True)  # Default: enabled
    task_name = config.get("task_name", "unknown_task")

    # Retry configuration
    retry_attempts = int(config.get("retry_attempts", 3))
    retry_min_wait = float(config.get("retry_min_wait", 1.0))
    retry_max_wait = float(config.get("retry_max_wait", 60.0))

    # Failure threshold configuration (default: 10% error rate)
    max_error_rate = float(config.get("max_error_rate", 0.1))

    # Pre-flight check configuration (default: enabled)
    preflight_check = config.get("preflight_check", True)

    # Structured outputs configuration
    response_format_example = config.get("response_format")
    score_field = config.get("score_field", "score")

    # Infer JSON schema once if response_format is defined
    json_schema = None
    if response_format_example:
        if not GENSON_AVAILABLE:
            eval_logger.warning(
                "response_format requires genson package. Install with: pip install genson. "
                "Falling back to text-based response."
            )
        else:
            json_schema = _infer_json_schema_from_example(
                response_format_example, name="llm_judge_response"
            )

    # Build metric key for results and details storage
    metric_key = f"llm_judge_{judge_name}" if judge_name else "llm_judge"
    judge_label = f"LLM Judge ({judge_name})" if judge_name else "LLM Judge"

    # Log configuration details
    eval_logger.info(f"=== {judge_label} Configuration ===")
    eval_logger.info(f"  Task: {task_name}")
    eval_logger.info(f"  Model: {model}")
    eval_logger.info(f"  API endpoint: {api_base or 'https://api.openai.com/v1 (default)'}")
    eval_logger.info(f"  Items to evaluate: {len(items)}")
    eval_logger.info(f"  Concurrency: {concurrency}")

    # Log retry settings
    if TENACITY_AVAILABLE and retry_attempts > 0:
        eval_logger.info(f"  Retry: ENABLED ({retry_attempts} attempts, {retry_min_wait}s-{retry_max_wait}s exponential backoff)")
    else:
        if not TENACITY_AVAILABLE:
            eval_logger.info(f"  Retry: DISABLED (tenacity not installed, run: pip install tenacity)")
        else:
            eval_logger.info(f"  Retry: DISABLED (retry_attempts=0)")

    eval_logger.info(f"  Max error rate: {max_error_rate:.1%} (evaluation fails if exceeded)")

    if json_schema:
        eval_logger.info(f"  Structured outputs: ENABLED (score_field='{score_field}')")
        eval_logger.info(f"  Inferred JSON schema: {json.dumps(json_schema, indent=2)}")
    else:
        eval_logger.info(f"  Structured outputs: DISABLED (using text-based 'Score: X.XX' parsing)")

    # Run pre-flight check to verify API connectivity before batch evaluation
    if preflight_check:
        _preflight_check_llm_judge(
            api_base=api_base,
            api_key=api_key,
            model=model,
            judge_label=judge_label,
        )
    else:
        eval_logger.info(f"  Pre-flight check: SKIPPED (preflight_check=false)")

    # Log first formatted prompt as example
    if items:
        first_item = items[0]
        example_prompt = _render_llm_judge_prompt(
            prompt_template=prompt_template,
            prediction=str(first_item[1]),
            reference=str(first_item[0]) if first_item[0] is not None else None,
            doc=first_item[2],
            config=config,
        )
        eval_logger.info(f"  Example prompt (first item):\n{'=' * 40}\n{example_prompt}\n{'=' * 40}")

    eval_logger.info(f"Starting {judge_label} evaluation...")

    def evaluate_item(idx, item):
        reference, prediction, doc = item[0], item[1], item[2]
        result = _call_llm_judge_single(
            prediction=str(prediction),
            reference=str(reference) if reference is not None else None,
            doc=doc,
            prompt_template=prompt_template,
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            config=config,
            json_schema=json_schema,
            score_field=score_field,
            retry_attempts=retry_attempts,
            retry_min_wait=retry_min_wait,
            retry_max_wait=retry_max_wait,
        )
        result["idx"] = idx
        return result

    from tqdm import tqdm

    # Store results indexed by position to maintain order
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(evaluate_item, i, item): i for i, item in enumerate(items)
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(items),
            desc=judge_label,
            unit="item",
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                eval_logger.error(f"Error in LLM judge evaluation: {e}")
                results[idx] = {
                    "idx": idx,
                    "score": 0.0,
                    "judgment_raw": None,
                    "explanation": None,
                    "formatted_prompt": None,
                    "prediction": str(items[idx][1]),
                    "reference": str(items[idx][0]) if items[idx][0] else None,
                    "error": str(e),
                }

    # Check error rate against threshold
    error_count = sum(1 for r in results if r is not None and r.get("error") is not None)
    total_count = len(results)
    error_rate = error_count / total_count if total_count > 0 else 0.0

    eval_logger.info(f"{judge_label} completed: {total_count - error_count}/{total_count} successful ({error_count} errors, {error_rate:.1%} error rate)")

    if error_rate > max_error_rate:
        error_msg = (
            f"{judge_label} FAILED: Error rate {error_rate:.1%} exceeds threshold {max_error_rate:.1%}. "
            f"{error_count}/{total_count} API calls failed. "
            f"Check logs for details or increase max_error_rate in config."
        )
        eval_logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Store detailed results for deferred saving via EvaluationTracker
    # This ensures proper file paths with model_name_sanitized
    # Use composite key to support multiple judges per task (e.g., "task_name/accuracy")
    if save_details:
        global _llm_judge_pending_details
        details_key = f"{task_name}/{judge_name}" if judge_name else task_name
        _llm_judge_pending_details[details_key] = {
            "judge_name": judge_name,
            "judge_model": model,
            "results": results,
        }

    scores = [r["score"] for r in results if r is not None]

    if len(scores) == 0 or all(np.isnan(s) if isinstance(s, float) else False for s in scores):
        return np.nan

    return np.nanmean(scores)


@register_metric(
    metric="llm_judge",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="llm_judge",
)
def llm_judge_fn(items):
    """
    Passthrough metric function that collects items for batch evaluation.

    The actual LLM judge API calls are made in the llm_judge_agg aggregation
    function, which processes all items concurrently.

    Items passed to this function should be tuples of:
    (reference, prediction, doc, config)

    Example YAML configuration with Jinja template:
    ```yaml
    metric_list:
      - metric: llm_judge
        aggregation: llm_judge
        higher_is_better: true
        prompt_template: |
          Evaluate this translation:

          Source: {{ doc.source_text }}
          Reference: {{ reference }}
          Translation: {{ prediction }}

          Score from 0-10:
        api_base: https://api.openai.com/v1
        model: gpt-4
        temperature: 0.0
        concurrency: 32
    ```

    Template variables available:
    - prediction: The model's generated response
    - reference: The gold/target answer
    - doc: The full document dict
    - All fields from doc are also available directly (e.g., {{ source_text }})
    """
    return items
