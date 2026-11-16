from __future__ import annotations

import logging
import re
from functools import cache
from typing import TYPE_CHECKING

from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    4096,
]

# The keys a model name can arrive under. RULER tasks are handed the parsed
# `--model_args`, and each backend names the model differently: `pretrained`
# for local HuggingFace models, `model` for the API backends.
TOKENIZER_ARG_KEYS = ("tokenizer", "pretrained", "model")


def resolve_tokenizer_name(kwargs: dict) -> str | None:
    """Pick the tokenizer name out of the model args a synthetic task is given.

    Returns the first non-empty string among `tokenizer`, `pretrained` and
    `model`, or None if there is none. Returning None rather than a container
    matters: `get_tokenizer` is decorated with `functools.cache`, which hashes
    its arguments before the body runs, so an unhashable default such as `{}`
    raises `TypeError: unhashable type: 'dict'` and the assertion below never
    gets the chance to say what is actually wrong.
    """
    for key in TOKENIZER_ARG_KEYS:
        value = kwargs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast:
    pretrained = tokenizer or pretrained
    assert pretrained, (
        "No tokenizer or pretrained provided. RULER tasks generate their own "
        "data and need a tokenizer to size it to each target length; pass one "
        "with `--model_args tokenizer=<name>`."
    )
    eval_logger.info("Using tokenizer %s for synthetic tasks.", pretrained)
    tok = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

    # Every synthetic task sizes its prompts by token count, so a tokenizer that
    # under-reports lengths sends the samplers into unbounded loops. A repo that
    # ships no tokenizer files (a GGUF-only repo, say) still loads under
    # `trust_remote_code=True` as an empty-vocab tokenizer that encodes anything
    # to 0 tokens -- reject it here rather than in each sampler.
    if len(tok("The quick brown fox jumps over the lazy dog.").input_ids) == 0:
        raise ValueError(
            f"Tokenizer '{pretrained}' encodes a non-empty string to 0 tokens "
            f"(vocab_size={getattr(tok, 'vocab_size', '?')}), so sequence lengths "
            "cannot be measured. Point the synthetic tasks at a repo that ships "
            'tokenizer files, e.g. --metadata=\'{"tokenizer": "Qwen/Qwen3-0.6B"}\'.'
        )
    return tok


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
        for pred, ref in zip(preds, refs, strict=False)
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
        for pred, ref in zip(preds, refs, strict=False)
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
