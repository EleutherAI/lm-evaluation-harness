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
import itertools
import random
import string

import datasets
import numpy as np
import transformers
from scipy.special import zeta
from tqdm import tqdm

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer


CONFIG = {
    "tokens_to_generate": 50,
    "template": """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
    "answer_prefix": """ Answer: According to the coded text above, the three most frequently appeared words are:""",
}


SEED = 42
TEMPLATE = CONFIG["template"] + CONFIG["answer_prefix"]


def generate_input_output(
    max_len: int,
    tokenizer: "transformers.PreTrainedTokenizerFast",
    num_words=-1,
    coded_wordlen=6,
    vocab_size=2000,
    incremental=10,
    alpha=2.0,
) -> tuple[str, list[str], int]:
    # generate vocab
    vocab = [
        "".join(random.choices(string.ascii_lowercase, k=coded_wordlen))
        for _ in range(vocab_size)
    ]
    while len(set(vocab)) < vocab_size:
        vocab.append("".join(random.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(list(set(vocab)))
    random.Random(SEED).shuffle(vocab)
    vocab[0] = "..."  # treat the top ranked as noise

    # sample words
    template = TEMPLATE

    def gen_text(num_words):
        k = np.arange(1, len(vocab) + 1)
        sampled_cnt = num_words * (k**-alpha) / zeta(alpha)
        sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
        sampled_words = [x for wlst in sampled_words for x in wlst]
        random.Random(SEED).shuffle(sampled_words)
        return template.format(context=" ".join(sampled_words), query=""), vocab[1:4]

    if num_words > 0:
        num_words = num_words
        text, answer = gen_text(num_words)
        while len(tokenizer(text).input_ids) > max_len:
            num_words -= incremental
            text, answer = gen_text(num_words)
    else:
        num_words = max_len // coded_wordlen  # init
        text, answer = gen_text(num_words)
        while len(tokenizer(text).input_ids) < max_len:
            num_words += incremental
            text, answer = gen_text(num_words)
        num_words -= incremental
    text, answer = gen_text(num_words)
    return text, answer, num_words


def sys_kwext(
    tokenizer: "transformers.PreTrainedTokenizerFast",
    max_seq_length: int,
    num_samples: int = 500,
    vocab_size: int = -1,
    coded_wordlen: int = 6,
    alpha: float = 2.0,
    tokens_to_generate: int = 50,
    remove_newline_tab: bool = False,
) -> list[dict]:
    write_jsons = []
    tokens_to_generate = tokens_to_generate

    vocab_size = max_seq_length // 50 if vocab_size == -1 else vocab_size

    # get number of words
    input_max_len = max_seq_length
    _, _, num_example_words = generate_input_output(
        input_max_len,
        tokenizer,
        coded_wordlen=coded_wordlen,
        vocab_size=vocab_size,
        incremental=input_max_len // 32,
        alpha=alpha,
    )
    # Generate samples
    for index in tqdm(
        range(num_samples), desc=f"Generating FWE Samples | {max_seq_length}"
    ):
        # construct input
        input_max_len = max_seq_length
        input_text, answer, _ = generate_input_output(
            input_max_len,
            tokenizer,
            num_words=num_example_words,
            coded_wordlen=coded_wordlen,
            vocab_size=vocab_size,
            incremental=input_max_len // 32,
            alpha=alpha,
        )

        length = len(tokenizer(input_text).input_ids) + tokens_to_generate

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": input_text[: input_text.rfind(CONFIG["answer_prefix"])].strip(),
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": CONFIG["answer_prefix"].strip(),
        }
        write_jsons.append(formatted_output)

    return write_jsons


def get_dataset(pretrained, max_seq_length=None, **kwargs):
    tokenizer = get_tokenizer(pretrained)
    write_jsons = sys_kwext(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    return write_jsons


def fwe_download(**kwargs):
    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
    df = (
        get_dataset(pretrained, max_seq_length=seq)
        for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    )

    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }
