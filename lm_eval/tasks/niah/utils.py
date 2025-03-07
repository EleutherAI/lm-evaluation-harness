import logging
import random
import re
from functools import cache
from typing import TYPE_CHECKING, Literal, Union

import datasets
import numpy as np
import wonderwords
from tqdm import tqdm
from transformers import AutoTokenizer


eval_logger = logging.getLogger(__name__)


# Words
r = wonderwords.RandomWord()

nouns = r._categories["nouns"]
adjs = r._categories["adjectives"]
verbs = r._categories["verbs"]
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
WORDS = sorted(list(set(words)))
DEFAULT_SEQ_LENGTHS = (4096,)


if TYPE_CHECKING:
    import transformers


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using Tokenizer for synthetic task: {pretrained}")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


@cache
def get_haystack(
    type_haystack: Literal["essay", "repeat", "needle"],
) -> Union[list[str], str]:
    if type_haystack == "essay":
        assert False, "TODO"
        essay = datasets.load_dataset("baber/paul_graham_essays", split="train")["text"]
        essay = " ".join(essay)
        haystack = re.sub(r"\s+", " ", essay).split(" ")
    elif type_haystack == "repeat":
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    else:
        raise NotImplementedError(f"{type_haystack} is not implemented.")
    return haystack


def create_samples(
    tokenizer, max_seq_length: int, samples, depth: int, max_gen_toks: int = 128
) -> list[dict]:
    task_description = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards."
    haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    haystack_length = len(tokenizer.encode(haystack, add_special_tokens=False))
    res = []
    for _ in range(samples):
        key = random.choice(WORDS)
        value = random.randint(10000, 50000)
        needle = f"The special magic number for {key} is {value}"
        question = f"Question: What is the special magic number for {key}\n"
        gen_prefix = f"Answer: The special magic number for {key} is"

        # Calculate token lengths for all components
        reserved = (
            sum(
                len(x)
                for x in tokenizer(
                    [task_description, needle, question, gen_prefix]
                ).input_ids
            )
            + max_gen_toks
        )

        available_tokens = max_seq_length - reserved

        num_haystacks = max(0, available_tokens // ((haystack_length + 1) + 1))

        # Create output with appropriate number of haystacks
        output = [task_description]
        for _ in range(num_haystacks):
            output.append(haystack)

        # Calculate insertion position for needle based on depth percentage
        if num_haystacks > 0:
            depth_position = max(
                1, min(len(output), int((num_haystacks + 1) * (depth / 100)))
            )
            output.insert(depth_position, needle)
        else:
            # If we can't fit any haystacks, just put the needle after task description
            output.append(needle)

        # Add the question at the end
        output.append(question)

        # Construct the final input
        input_text = "\n".join(output)

        # Verify the length is within limits
        total_length = len(tokenizer.encode(input_text + gen_prefix))
        assert total_length + max_gen_toks <= max_seq_length

        res.append(
            {
                "input": input_text,
                "key": str(key),
                "value": str(value),
                "gen_prefix": gen_prefix,
                "context_length": total_length,
                "max_length": max_seq_length,
                "depth": depth,
            }
        )
    return res


def create_all_samples(
    tokenizer, seq_lengths: list[int], depths: list[int] = None, samples: int = None
) -> dict[str, datasets.Dataset]:
    if depths is None:
        depths = [10, 25, 50, 75, 90, 100]
    if samples is None:
        samples = 1
    res = []
    for seq_length in tqdm(seq_lengths, desc="Generating samples"):
        for depth in depths:
            res.extend(create_samples(tokenizer, seq_length, samples, depth))
    return {"test": datasets.Dataset.from_list(res, split=datasets.Split.TEST)}


def create_dataset(**kwargs) -> dict[str, datasets.Dataset]:
    seq_lengths = kwargs.pop(
        "max_seq_lengths", np.linspace(27540, 62940, 19, dtype=int)
    )
    tokenizer = get_tokenizer(**kwargs)
    return create_all_samples(tokenizer, seq_lengths)


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


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results[0])
    score = string_match_all([pred], [doc["value"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return 0.0
    return sum(res) / len(res)
