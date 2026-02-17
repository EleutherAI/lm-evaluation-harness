"""Per-item metric functions for loglikelihood / multiple-choice / generate_until tasks.

Each function receives ``(references, predictions)`` and returns a single scalar
that will be collected across documents and fed to the registered aggregation
function (typically ``mean``).
"""

from __future__ import annotations

import re
import string
from typing import TYPE_CHECKING

import numpy as np


NAT_TO_BIT = 1.0 / np.log(2.0)

from lm_eval.api.registry import register_metric


if TYPE_CHECKING:
    from lm_eval._types import LLResults


# ---------------------------------------------------------------------------
# Accuracy variants
# ---------------------------------------------------------------------------
def _multiple_targets(_target: int | list[int], _result: int):
    _target = [
        t for t in (_target if isinstance(_target, list) else [_target]) if t != -100
    ]
    return int(any(_result == t for t in _target))


@register_metric(
    metric="acc",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_fn(
    references: int | list[int], predictions: LLResults, multiple_targets=False
) -> int:
    """Accuracy.

    For multiple-choice (multiple lls): 1 if argmax(lls) matches gold.
    For single loglikelihood (one ll): 1 if the continuation was decoded greedily.
    """

    if len(predictions.lls) == 1:
        # Plain loglikelihood: acc = greedy decode match
        return int(predictions.is_greedy[0])
    pred = int(np.argmax(predictions.lls))
    if multiple_targets:
        return _multiple_targets(predictions.targets, pred)
    assert not isinstance(references, list), (
        "Multiple targets not supported for acc metric without multiple_targets=True"
    )
    return int(pred == int(references))


@register_metric(
    metric="acc_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_norm_fn(
    references: int | list[int], predictions: LLResults, multiple_targets=False
) -> int:
    """Character-length-normalised accuracy: picks the choice with the highest ``ll / char_len``."""
    pred = int(np.argmax(np.array(predictions.lls) / np.array(predictions.char_len())))
    if multiple_targets:
        return _multiple_targets(predictions.targets, pred)
    assert not isinstance(references, list), (
        "Multiple targets not supported for acc metric without multiple_targets=True"
    )
    return int(pred == int(references))


@register_metric(
    metric="acc_bytes",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def acc_bytes_fn(
    references: int | list[int], predictions: LLResults, multiple_targets=False
) -> int:
    """Byte-length-normalised accuracy: picks the choice with the highest ``ll / byte_len``."""
    pred = int(np.argmax(np.array(predictions.lls) / np.array(predictions.byte_len())))
    if multiple_targets:
        return _multiple_targets(references, pred)
    assert not isinstance(references, list), (
        "Multiple targets not supported for acc metric without multiple_targets=True"
    )
    return int(pred == int(references))


@register_metric(
    metric="acc_mutual_info",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def acc_mutual_info_fn(
    references: int | list[int], predictions: LLResults, multiple_targets=False
) -> int:
    """Mutual-information-weighted accuracy: picks the choice with the highest ``ll - ll_unconditional``."""
    pred = int(np.argmax(predictions.lls_mutual_info))
    if multiple_targets:
        return _multiple_targets(references, pred)
    assert not isinstance(references, list), (
        "Multiple targets not supported for acc metric without multiple_targets=True"
    )
    return int(pred == int(references))


# ---------------------------------------------------------------------------
# Exact-match (greedy) for multiple-choice
# ---------------------------------------------------------------------------


@register_metric(
    metric="exact_match_mc",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="mean",
)
def exact_match_mc_fn(references: int | list[int], predictions: LLResults) -> int:
    """1 if the gold completion was decoded greedily (every token was argmax), else 0."""
    if isinstance(references, list):
        return int(
            any(predictions.is_greedy[i] if i != -100 else False for i in references)
        )
    if references == -100:
        return 0
    return int(predictions.is_greedy[int(references)])


# ---------------------------------------------------------------------------
# Log-probability / bits-per-byte metrics
# ---------------------------------------------------------------------------


@register_metric(
    metric="bpb",
    higher_is_better=False,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def bpb_fn(references: int, predictions: LLResults) -> float:
    """Bits-per-byte of the gold completion: ``-ll[gold] / byte_len[gold] * NAT_TO_BIT``.

    Lower is better — measures how many bits the model needs per byte of the
    correct answer.
    """
    return (
        -predictions.lls[references] / predictions.byte_len()[references]
    ) * NAT_TO_BIT


@register_metric(
    metric="logprob",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def logprob_fn(references: int, predictions: LLResults) -> float:
    """Raw log-probability of the gold completion (in nats)."""
    return float(predictions.lls[references])


@register_metric(
    metric="choice_logprob",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def choice_logprob_fn(references: int, predictions: LLResults) -> float:
    """Log-probability of the gold choice under a softmax over all choices.

    Equals ``ll[gold] - logsumexp(ll)``, i.e. treating the raw log-likelihoods
    as logits and returning the log-probability assigned to the correct answer.
    """
    lls = np.array(predictions.lls)
    return float(lls[references] - np.logaddexp.reduce(lls))


@register_metric(
    metric="choice_prob_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def choice_prob_norm_fn(references: int, predictions: LLResults) -> float:
    """Length-normalised probability of the gold choice.

    Each choice is weighted by its nats-per-byte (``ll / byte_len``), then a
    softmax is applied. Returns the probability mass on the correct answer.
    This compensates for longer completions receiving lower raw log-likelihoods.
    """
    lls = np.array(predictions.lls)
    byte_len = np.array(predictions.byte_len(), dtype=float)
    log_weights = lls / byte_len
    return float(np.exp(log_weights[references] - np.logaddexp.reduce(log_weights)))


@register_metric(
    metric="choice_logprob_norm",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice"],
    aggregation="mean",
)
def choice_logprob_norm_fn(references: int, predictions: LLResults) -> float:
    """Log of the length-normalised probability of the gold choice.

    Equivalent to ``log(choice_prob_norm)`` but computed in log-space for
    numerical stability.
    """
    lls = np.array(predictions.lls)
    byte_len = np.array(predictions.byte_len(), dtype=float)
    log_weights = lls / byte_len
    return float(log_weights[references] - np.logaddexp.reduce(log_weights))


# ---------------------------------------------------------------------------
# Perplexity (passthrough ll for corpus-level aggregation)
# ---------------------------------------------------------------------------


@register_metric(
    metric="perplexity",
    higher_is_better=False,
    output_type="loglikelihood",
    aggregation="perplexity",
)
def perplexity_fn(references: int | list[int], predictions: LLResults) -> float:
    """Passthrough of the gold log-likelihood for corpus-level perplexity aggregation."""
    return predictions.lls[predictions.target]


# ---------------------------------------------------------------------------
# generate_until: exact match
# ---------------------------------------------------------------------------

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


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)


# ---------------------------------------------------------------------------
# Bypass (no-op passthrough)
# ---------------------------------------------------------------------------


@register_metric(
    metric="bypass",
    higher_is_better=True,
    output_type=["loglikelihood", "multiple_choice", "generate_until"],
    aggregation="bypass",
)
def bypass(items):
    return None
