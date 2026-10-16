"""Evalita NER macro F1 averages its three declared entity categories."""

import pytest

from lm_eval.tasks.evalita_llm.metrics import _aggreg_ner
from lm_eval.tasks.evalita_llm.utils import ner_process_results


def confusion_count_macro(predictions, references):
    scores = []
    for label in (0, 1, 2):  # PER, LOC, ORG
        tp = sum(
            p == label and r == label
            for p, r in zip(predictions, references, strict=True)
        )
        fp = sum(
            p == label and r != label
            for p, r in zip(predictions, references, strict=True)
        )
        fn = sum(
            p != label and r == label
            for p, r in zip(predictions, references, strict=True)
        )
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / 3


@pytest.mark.parametrize(
    "predictions, references",
    [
        ([0, 0], [0, 1]),
        ([0, 1], [0, 1]),
        ([2, 2], [0, 2]),
        ([0], [0]),
        ([3], [3]),
        ([0, 1, 2, 3], [0, 1, 2, 3]),
        ([0, 3, 2, 0], [0, 1, 2, 3]),
    ],
)
def test_macro_score_uses_entity_labels_not_last_observed_class(
    predictions, references
):
    expected = confusion_count_macro(predictions, references)
    assert _aggreg_ner([(predictions, references)]) == pytest.approx(expected)


def test_no_entity_classification_does_not_add_an_outside_class_to_macro():
    entities = [
        {"entity_text": "Mario", "type": "PER"},
        {"entity_text": "Roma", "type": "LOC"},
    ]
    item = ner_process_results({"entities": entities}, ["Mario$PER, Roma$LOC"])["f1"]
    assert _aggreg_ner([item]) == pytest.approx(2 / 3)


def test_corpus_partition_does_not_change_the_confusion_counts():
    rows = [([0, 3], [0, 1]), ([2, 0], [2, 3])]
    flat_pred = [label for predictions, _ in rows for label in predictions]
    flat_ref = [label for _, references in rows for label in references]
    assert _aggreg_ner(rows) == pytest.approx(
        confusion_count_macro(flat_pred, flat_ref)
    )


@pytest.mark.parametrize(
    "output, predictions, references",
    [
        ("Luigi$PER,Roma$LOC,Fiat$ORG", [3, 1, 2], [0, 1, 2]),
        (
            "Luigi$PER,Roma$LOC,Fiat$ORG,Milano$LOC",
            [3, 1, 2, 2],
            [0, 1, 2, 3],
        ),
        (
            "Roma$LOC,Fiat$ORG,x$LOC,y$LOC",
            [3, 1, 2, 2],
            [0, 1, 2, 3],
        ),
        (
            "Mario Rossi$PER,Fiat$ORG,x$LOC,y$LOC",
            [0, 3, 2, 2],
            [0, 1, 2, 3],
        ),
        (
            "Mario Rossi$PER,Roma$LOC,x$LOC,y$LOC",
            [0, 1, 3, 2],
            [0, 1, 2, 3],
        ),
        (
            "Mario Rossi$PER,Roma$LOC,Fiat$ORG,Milano$LOC",
            [0, 1, 2, 2],
            [0, 1, 2, 3],
        ),
    ],
)
def test_unmatched_gold_entities_remain_misses_with_extra_predictions(
    output, predictions, references
):
    entities = [
        {"entity_text": "Mario Rossi", "type": "PER"},
        {"entity_text": "Roma", "type": "LOC"},
        {"entity_text": "Fiat", "type": "ORG"},
    ]
    item = ner_process_results({"entities": entities}, [output])["f1"]
    assert item == (predictions, references)
    assert _aggreg_ner([item]) == pytest.approx(
        confusion_count_macro(predictions, references)
    )


def test_unrelated_long_outputs_cannot_recover_missed_people_in_a_corpus():
    rows = [
        ner_process_results(
            {"entities": [{"entity_text": name, "type": "PER"}]},
            ["Roma$LOC,Fiat$ORG"],
        )["f1"]
        for name in ("Mario", "Luigi", "Giulia", "Anna")
    ]
    assert _aggreg_ner(rows) == 0.0
