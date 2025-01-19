import itertools
from typing import Generator

import datasets

from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack
from lm_eval.tasks.ruler.common_utils import SEQ_LENGTHS, get_tokenizer

TEMPLATE = """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"""


def download_dataset(df: Generator) -> dict[str, datasets.Dataset]:
    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }


# ruff: noqa
niah_single_1 = lambda **kwargs: download_dataset(
    generate_samples(
        get_haystack(type_haystack="repeat"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="repeat",
        type_needle_k="words",
        type_needle_v="numbers",
        TOKENIZER=get_tokenizer(**kwargs.get("metadata")),
    )
    for seq in SEQ_LENGTHS
)
# ruff: noqa
niah_single_2 = lambda **kwargs: download_dataset(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        TOKENIZER=get_tokenizer(**kwargs.get("metadata")),
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_single_3 = lambda **kwargs: download_dataset(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="uuids",
        TOKENIZER=get_tokenizer(**kwargs.get("metadata")),
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multikey_1 = lambda **kwargs: download_dataset(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_k=4,
        TOKENIZER=get_tokenizer(**kwargs.get("metadata")),
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multikey_2 = lambda **kwargs: download_dataset(
    generate_samples(
        get_haystack(type_haystack="needle"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="needle",
        type_needle_k="words",
        type_needle_v="numbers",
        TOKENIZER=get_tokenizer(**kwargs.get("metadata")),
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multikey_3 = lambda **kwargs: download_dataset(
    generate_samples(
        get_haystack(type_haystack="needle"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="needle",
        type_needle_k="uuids",
        type_needle_v="uuids",
        TOKENIZER=get_tokenizer(**kwargs.get("metadata")),
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multivalue = lambda **kwargs: download_dataset(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_v=4,
        TOKENIZER=get_tokenizer(**kwargs.get("metadata")),
    )
    for seq in SEQ_LENGTHS
)
# noqa
niah_multiquery = lambda **kwargs: download_dataset(
    generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=seq,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_q=4,
        TOKENIZER=get_tokenizer(**kwargs.get("metadata")),
    )
    for seq in SEQ_LENGTHS
)
