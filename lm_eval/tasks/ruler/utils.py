# noqa
import itertools
import json
import os
import re
from functools import partial
from typing import Literal

import datasets
from transformers import AutoTokenizer

from lm_eval.tasks.ruler.essays import get_essays
from lm_eval.tasks.ruler.prepare import generate_samples


TOKENIZER = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER"))
TEMPLATE = """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"""

SEQ_LENGTHS = (
    131072,
    65536,
    32768,
    16384,
    8192,
    4096,
)

NUM_SAMPLES = 500
REMOVE_NEWLINE_TAB = ""
STOP_WORDS = ""
RANDOM_SEED = 42


def get_haystack(type_haystack: Literal["essay", "repeat", "needle"]):
    NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
    if type_haystack == "essay":
        essay = get_essays()["text"]
        # essay = json.load(open(essay))["text"]
        haystack = re.sub(r"\s+", " ", essay).split(" ")
    elif type_haystack == "repeat":
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    elif type_haystack == "needle":
        haystack = NEEDLE
    else:
        raise NotImplementedError(f"{type_haystack} is not implemented.")
    return haystack


def flatten(df):
    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }


# ruff: noqa
niah_single_1 = lambda: flatten(
    generate_samples(
        get_haystack(type_haystack="repeat"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="repeat",
        type_needle_k="words",
        type_needle_v="numbers",
    )
    for seq in SEQ_LENGTHS
)
# ruff: noqa
niah_single_2 = lambda: flatten(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_single_3 = lambda: flatten(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="uuids",
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multikey_1 = lambda: flatten(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_k=4,
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multikey_2 = lambda: flatten(
    generate_samples(
        get_haystack(type_haystack="needle"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="needle",
        type_needle_k="words",
        type_needle_v="numbers",
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multikey_3 = lambda: flatten(
    generate_samples(
        get_haystack(type_haystack="needle"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="needle",
        type_needle_k="uuids",
        type_needle_v="uuids",
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multivalue = lambda: flatten(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_v=4,
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multiquery = lambda: flatten(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_q=4,
    )
    for seq in SEQ_LENGTHS
)


def postprocess_pred(predict_str: str):
    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r"[\x00-\x1f]")
    predict_str = np_pattern.sub("\n", predict_str).strip()

    return predict_str


def process_results(doc, results):
    metrics = {str(length): -1.0 for length in SEQ_LENGTHS}
    input_len = doc["max_length"]
    acc = 1.0 if postprocess_pred(results[0]) in doc["input"] else 0.0
    metrics[str(next(length for length in SEQ_LENGTHS if input_len <= length))] = acc
    return metrics


def aggregate_metrics(metrics):
    return {
        length: sum(metric[length] for metric in metrics) / len(metrics)
        for length in SEQ_LENGTHS
    }
