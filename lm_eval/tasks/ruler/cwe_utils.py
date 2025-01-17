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


import random

import wonderwords
from tqdm import tqdm


RNG = random.Random(42)
TEMPLATE = ""
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


def generate_input_output(num_words, max_seq_length, freq_cw=30, freq_ucw=3, num_cw=10):
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

    return input_example + "\n" + input_text, answer


def sys_word_pair_random(
    num_samples: int,
    max_seq_length: int,
    TOKENIZER=None,
    incremental: int = 10,
    remove_newline_tab=False,
    tokens_to_generate=120,
):
    assert TOKENIZER is not None, "Tokenizer is not provided."
    write_jsons = []
    tokens_to_generate = tokens_to_generate

    # Find the perfect num_words
    num_words = incremental

    total_tokens = 0
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer = generate_input_output(num_words, max_seq_length)
        # Calculate the number of tokens in the example
        total_tokens = len(
            TOKENIZER(
                input_text
                + " "
                + " ".join([f"{i + 1}. {word}" for i, word in enumerate(answer)])
            )
        )
        print(
            f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Words: {num_words}"
        )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_words -= incremental
            break

        num_words += incremental
        if num_words > len(WORDS):
            num_words = len(WORDS)
            break

    print("num_words:", num_words)

    # Generate samples
    for index in tqdm(range(num_samples)):
        used_words = num_words
        while True:
            try:
                input_text, answer = generate_input_output(used_words)
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_words > incremental:
                    used_words -= incremental

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
        }
        write_jsons.append(formatted_output)

    return write_jsons
