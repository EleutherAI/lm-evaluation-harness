import re
from functools import cache
from typing import TYPE_CHECKING, Union

from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


DEFAULT_SEQ_LENGTHS = (
    # 131072,
    # 65536,
    # 32768,
    16384,
    8192,
    4096,
)


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    print("using tokenizer ", pretrained)
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def postprocess_pred(predict_str: str) -> str:
    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r"[\x00-\x1f]")
    predict_str = np_pattern.sub("\n", predict_str).strip()

    return predict_str


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results[0])
    score = string_match_all([pred], [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results[0])
    score = string_match_part([pred], [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return 0.0
    return sum(res) / len(res)
