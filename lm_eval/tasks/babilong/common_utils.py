import logging
import re
from functools import cache
from typing import TYPE_CHECKING, Union

import datasets
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for babilong tasks.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def load_dataset(**kwargs):
    config_name = kwargs.get("max_seq_lengths", "0k")

    # Get specific qa split
    qa_split = kwargs.get("qa_split")

    eval_logger.info(
        f"Loading babilong dataset: max_seq_lengths={config_name}, split={qa_split}"
    )
    dataset = datasets.load_dataset(
        "RMT-team/babilong-1k-samples", name=config_name, split=qa_split
    )
    return {qa_split: dataset}


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    pred = postprocess_pred(results)
    target = doc.get("target", "").strip()

    # String match
    score = 1.0 if target.lower() in pred[0].lower() else 0.0

    return {"acc": score}
