"""Shared preprocessing for nortruthfulqa_mc (both languages).

The dataset lists the single correct answer first in mc1_targets, so without
this step the hybrid and mcf formations would always present it as the first
option.  process_docs shuffles the options of every document; the shuffle is
seeded by the document id, so the order is reproducible across runs and
independent of --limit, few-shot sampling and the document order.  The gold
index is wherever label 1 lands, see mc1_targets.labels.
"""

import random

import datasets


def shuffle_choices(doc):
    targets = doc["mc1_targets"]
    order = list(range(len(targets["choices"])))
    random.Random(doc["id"]).shuffle(order)
    return {
        "mc1_targets": {
            "choices": [targets["choices"][i] for i in order],
            "labels": [targets["labels"][i] for i in order],
        }
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(shuffle_choices)
