"""Utilities for the OLMES Basic Skills task family."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from lm_eval.api.samplers import ContextSampler


if TYPE_CHECKING:
    from collections.abc import Sequence

    import datasets


CHOICE_LABELS = "ABCDEFGHIJKLMNOPQ"


def shuffle_and_insert(
    values: Sequence[str], value: str, rnd: random.Random
) -> tuple[list[str], int]:
    """Reproduce the choice construction in the pinned OLMES task."""
    shuffled = list(values)
    rnd.shuffle(shuffled)
    insert_index = rnd.randint(0, len(shuffled))
    shuffled.insert(insert_index, value)
    return shuffled, insert_index


def _shuffle_doc(doc: dict[str, Any]) -> tuple[list[str], int]:
    return shuffle_and_insert(
        doc["wrong_answers"], doc["answer"], random.Random(doc["id"])
    )


def _process_rc_doc(doc: dict[str, Any]) -> dict[str, Any]:
    choices, gold = _shuffle_doc(doc)
    return {"question": doc["question"], "choices": choices, "gold": gold}


def make_mcq_prompt(question: str, choices: Sequence[str]) -> str:
    """Format an OLMES ``make_mcq_prompt`` with its default labels."""
    choices_text = "\n".join(
        f" {label}. {choice}"
        for label, choice in zip(CHOICE_LABELS, choices, strict=False)
    )
    return f"{question}\n{choices_text}\nAnswer:"


def _process_mc_doc(doc: dict[str, Any]) -> dict[str, Any]:
    choices, gold = _shuffle_doc(doc)
    if len(choices) > len(CHOICE_LABELS):
        raise ValueError(
            f"Basic Skills example {doc['id']!r} has {len(choices)} choices; "
            f"the pinned OLMES task supports at most {len(CHOICE_LABELS)}"
        )
    return {
        "question": make_mcq_prompt(doc["question"], choices),
        "choices": list(CHOICE_LABELS[: len(choices)]),
        "gold": gold,
    }


def process_rc_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_rc_doc)


def process_mc_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_mc_doc)


class OLMESContextSampler(ContextSampler):
    """Match OLMES's stateless, per-evaluation-document few-shot sampling."""

    def __init__(self, *args: Any, rnd: int | None = None, **kwargs: Any) -> None:
        self.seed = rnd
        super().__init__(*args, rnd=rnd, **kwargs)

    def set_rnd(self, rnd: int | None):
        self.seed = rnd
        return super().set_rnd(rnd)

    def sample(
        self,
        n: int,
        eval_doc: dict[str, Any] | None = None,
        df: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Sequence[dict[str, Any]]:
        if n < 0:
            raise ValueError("number of few-shot examples must be non-negative")
        if n == 0:
            return []
        if df is not None:
            self.replace_df(df)

        pool = self.fewshot_docs()
        sample_size = n + 1 if eval_doc is not None else n
        sampled = random.Random(self.seed).sample(pool, sample_size)
        if eval_doc is not None:
            sampled = list(self.rm_eval_doc(eval_doc, sampled, n))

        if len(sampled) != n:
            raise ValueError(f"sampled {len(sampled)} few-shot examples, expected {n}")
        return sampled
