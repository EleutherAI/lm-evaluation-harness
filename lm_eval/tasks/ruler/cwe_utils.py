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

import datasets
import wonderwords
from tqdm import tqdm

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer


CONFIG = {
    "tokens_to_generate": 120,
    "template": """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?""",
    "answer_prefix": """ Answer: The top 10 words that appear most often in the list are:""",
}

RNG = random.Random(42)
TEMPLATE = CONFIG["template"] + CONFIG["answer_prefix"]


r = wonderwords.RandomWord()
WORDS = sorted(
    list(
        set([item for x in ["noun", "adjective", "verb"] for item in r._categories[x]])
    )
)
RNG.shuffle(WORDS)


def get_example(num_words, common_repeats=30, uncommon_repeats=3, common_nums=10):
    word_list_full = random.sample(WORDS, num_words)
    common, uncommon = word_list_full[:common_nums], word_list_full[common_nums:]
    word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
    RNG.shuffle(word_list)

    # Formatting the word list as "1. word1 2. word2 3. word3 ..."
    context = " ".join([f"{i + 1}. {word}" for i, word in enumerate(word_list)])

    return context, common


def generate_input_output(
    num_words: int,
    max_seq_length: int,
    freq_cw: int = 30,
    freq_ucw: int = 3,
    num_cw: int = 10,
):
    if max_seq_length < 4096:
        context_example, answer_example = get_example(20, 3, 1, num_cw)
        context, answer = get_example(num_words, 6, 1, num_cw)
    else:
        context_example, answer_example = get_example(40, 10, 3, num_cw)
        context, answer = get_example(num_words, freq_cw, freq_ucw, num_cw)

    template = TEMPLATE

    input_example = template.format(
        context=context_example,
        query="",
    ) + " ".join([f"{i + 1}. {word}" for i, word in enumerate(answer_example)])

    input_text = template.format(
        context=context,
        query="",
    )

    return input_example, input_text, answer


def sys_word_pair_random(
    num_samples: int,
    max_seq_length: int,
    tokenizer=None,
    incremental: int = 10,
    remove_newline_tab=False,
    tokens_to_generate=120,
):
    assert tokenizer is not None, "Tokenizer is not provided."
    write_jsons = []
    tokens_to_generate = tokens_to_generate

    # Find the perfect num_words
    num_words = incremental

    total_tokens = 0
    while total_tokens + tokens_to_generate < max_seq_length:
        input_example, input_text, answer = generate_input_output(
            num_words, max_seq_length
        )
        # Calculate the number of tokens in the example
        total_tokens = len(
            tokenizer(
                input_example
                + "\n"
                + input_text
                + " "
                + " ".join([f"{i + 1}. {word}" for i, word in enumerate(answer)])
            ).input_ids
        )
        # print(
        #     f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Words: {num_words}"
        # )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_words -= incremental
            break

        num_words += incremental
        if num_words > len(WORDS):
            num_words = len(WORDS)
            break

    # print("num_words:", num_words)

    # Generate samples
    for index in tqdm(
        range(num_samples), desc=f"Generating CWE Samples | {max_seq_length}"
    ):
        used_words = num_words
        while True:
            try:
                input_example, input_text, answer = generate_input_output(
                    used_words, max_seq_length
                )
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:  # noqa: E722
                if used_words > incremental:
                    used_words -= incremental

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )
            input_example = " ".join(
                input_example.replace("\n", " ").replace("\t", " ").strip().split()
            )

        gen_prefix_index = input_text.rfind(CONFIG["answer_prefix"])
        input_text = input_text[:gen_prefix_index]
        formatted_output = {
            "index": index,
            "input": input_text.strip(),
            "input_example": input_example,
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": CONFIG["answer_prefix"].strip(),
        }
        write_jsons.append(formatted_output)

    return write_jsons


def get_dataset(pretrained, seq=None, **kwargs):
    tokenizer = get_tokenizer(pretrained)
    write_jsons = sys_word_pair_random(
        num_samples=500, max_seq_length=seq, tokenizer=tokenizer
    )
    return write_jsons


def get_cw_dataset(**kwargs):
    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
    df = (
        get_dataset(pretrained, seq=seq)
        for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    )

    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }
