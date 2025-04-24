# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import os
import random
import re
import uuid
from functools import lru_cache, cache
from typing import List, Union, Literal
import datasets

import numpy as np
from packaging.version import parse as parse_version
from importlib.metadata import version

from tqdm import tqdm

try:
    import wonderwords
    import nltk
    from nltk import sent_tokenize
except ImportError:
    raise ImportError(
        'Please install the `wonderwords` and `nltk` packages to run this script. You can install them with `pip install lm_eval["ruler"]` or`pip install wonderwords nltk`.'
    )


NUM_SAMPLES = 500
REMOVE_NEWLINE_TAB = ""
STOP_WORDS = ""
RANDOM_SEED = 42
# Define Needle/Haystack Format
NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."


# Words
r = wonderwords.RandomWord()

nouns = r._categories["nouns"]
adjs = r._categories["adjectives"]
verbs = r._categories["verbs"]
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
WORDS = sorted(list(set(words)))

# Positions
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

NLTK_MIN_VERSION = "3.9.1"
RANK = os.environ.get("LOCAL_RANK", "0")


@lru_cache(maxsize=1024)
def cached_sent_tokenize(text: str) -> List[str]:
    return sent_tokenize(text)


def download_nltk_resources():
    """Download 'punkt' if not already installed"""
    assert (nltk_version := parse_version(version("nltk"))) >= parse_version(
        NLTK_MIN_VERSION
    ), (
        f"`nltk` version {nltk_version} is not >= {NLTK_MIN_VERSION}. Please update `nltk` before proceeding--older versions are vulnerable to a remote code execution vulnerability."
    )

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if RANK == "0":
            nltk.download("punkt_tab")
            print("Downloaded punkt_tab on rank 0")


download_nltk_resources()


def generate_random_number(num_digits=7) -> str:
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))


def generate_random_word() -> str:
    word = random.choice(WORDS)
    return word


def generate_random_uuid() -> str:
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def generate_random(type_needle: str) -> str:
    if type_needle == "numbers":
        return generate_random_number()
    elif type_needle == "words":
        return generate_random_word()
    elif type_needle == "uuids":
        return generate_random_uuid()
    else:
        raise NotImplementedError(f"{type_needle} is not implemented.")


def generate_input_output(
    num_haystack: int,
    haystack: Union[list[str], str],
    *,
    type_haystack: str,
    num_needle_k: int,
    type_needle_k: str,
    num_needle_v: int,
    type_needle_v: str,
    template: str,
    num_needle_q: int = 1,
    random_seed: int = RANDOM_SEED,
) -> tuple[str, list[str], str]:
    NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        keys.append(generate_random(type_needle_k))
        value = []
        for _ in range(num_needle_v):
            value.append(generate_random(type_needle_v))
            needles.append(
                NEEDLE.format(
                    type_needle_v=type_needle_v,
                    key=keys[-1],
                    value=value[-1],
                )
            )
        values.append(value)

    random.Random(random_seed).shuffle(needles)

    # Context
    if type_haystack == "essay":
        assert isinstance(haystack, list)
        text = " ".join(haystack[:num_haystack])
        document_sents = cached_sent_tokenize(text.strip())
        insertion_positions = (
            [0]
            + sorted(
                [
                    int(len(document_sents) * (depth / 100))
                    for depth in random.sample(DEPTHS, len(needles))
                ]
            )
            + [len(document_sents)]
        )
        document_sents_list = []
        for i in range(1, len(insertion_positions)):
            last_pos = insertion_positions[i - 1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i - 1 < len(needles):
                document_sents_list.append(needles[i - 1])
        context = " ".join(document_sents_list)

    else:
        if type_haystack == "repeat":
            sentences = [haystack] * num_haystack
        elif type_haystack == "needle":
            sentences = [
                haystack.format(
                    type_needle_v=type_needle_v,
                    key=generate_random(type_needle_k),
                    value=generate_random(type_needle_v),
                )
                for _ in range(num_haystack)
            ]

        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)

    ## Query and Answer
    indices = random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = (
        ", ".join(queries[:-1]) + ", and " + queries[-1]
        if len(queries) > 1
        else queries[0]
    )

    template = template
    type_needle_v = type_needle_v
    if num_needle_q * num_needle_v == 1:
        template = template.replace("Some", "A")
        template = template.replace("are all", "is")
        template = template.replace("are", "is")
        template = template.replace("answers", "answer")
        type_needle_v = type_needle_v[:-1]  # remove "s"

    input_text = template.format(
        type_needle_v=type_needle_v,
        context=context,
        query=query,
    )

    return input_text, answers, query


def generate_samples(
    haystack,
    TOKENIZER=None,
    *,
    max_seq_length: int,
    type_haystack: str,
    type_needle_k: str,
    type_needle_v: str,
    template: str,
    num_samples: int = 500,
    tokens_to_generate: int = 128,
    num_needle_v: int = 1,
    num_needle_k: int = 1,
    num_needle_q=1,
    incremental: int = 500,
    remove_newline_tab: bool = False,
    random_seed: int = 42,
) -> list[dict]:
    assert TOKENIZER is not None, "TOKENIZER is not defined."
    num_needle_k = max(num_needle_k, num_needle_q)
    write_jsons = []
    tokens_to_generate = tokens_to_generate

    if type_haystack == "essay":
        incremental = 500
    elif type_haystack == "repeat":
        incremental = 25
    elif type_haystack == "needle":
        incremental = 25

    if type_haystack != "essay" and max_seq_length < 4096:
        incremental = 5

    num_haystack = incremental

    total_tokens = 0  # Track the total tokens generated for the first example
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer, query = generate_input_output(
            num_haystack,
            haystack,
            type_haystack=type_haystack,
            num_needle_k=num_needle_k,
            type_needle_k=type_needle_k,
            num_needle_v=num_needle_v,
            type_needle_v=type_needle_v,
            template=template,
            num_needle_q=num_needle_q,
            random_seed=random_seed,
        )
        # Calculate the number of tokens in the example
        total_tokens = len(TOKENIZER(input_text + " ".join(answer)).input_ids)
        if total_tokens + tokens_to_generate > max_seq_length:
            num_haystack -= incremental
            break

        if type_haystack == "essay" and num_haystack > len(haystack):
            num_haystack = len(haystack)
            break

        num_haystack += incremental

    # print("Num haystack:", num_haystack)

    # Generate samples
    for index in tqdm(
        range(num_samples),
        desc=f"Generating synthetic samples: {type_haystack} | {max_seq_length}",
    ):
        used_haystack = num_haystack
        while True:
            try:
                input_text, answer, query = generate_input_output(
                    used_haystack,
                    haystack,
                    type_haystack=type_haystack,
                    num_needle_k=num_needle_k,
                    type_needle_k=type_needle_k,
                    num_needle_v=num_needle_v,
                    type_needle_v=type_needle_v,
                    template=template,
                    num_needle_q=num_needle_q,
                    random_seed=random_seed,
                )
                length = len(TOKENIZER(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
                # ruff: noqa
            except:
                if used_haystack > incremental:
                    used_haystack -= incremental

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": f"The special magic {type_needle_v[:-1]} for {query} mentioned in the provided text is"
            if num_needle_q * num_needle_v == 1
            else f"The special magic {type_needle_v} for {query} mentioned in the provided text are",
        }
        if formatted_output["outputs"][0] not in formatted_output["input"]:
            assert False, (
                f"Needle not in input: {formatted_output}. Something went wrong."
            )
        write_jsons.append(formatted_output)
    return write_jsons


@cache
def get_haystack(
    type_haystack: Literal["essay", "repeat", "needle"],
) -> Union[list[str], str]:
    NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
    if type_haystack == "essay":
        essay = datasets.load_dataset("baber/paul_graham_essays", split="train")["text"]
        essay = " ".join(essay)
        haystack = re.sub(r"\s+", " ", essay).split(" ")
    elif type_haystack == "repeat":
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    elif type_haystack == "needle":
        haystack = NEEDLE
    else:
        raise NotImplementedError(f"{type_haystack} is not implemented.")
    return haystack
