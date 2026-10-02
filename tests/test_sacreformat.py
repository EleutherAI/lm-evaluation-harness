"""Corpus translation metrics must retain uneven numbers of gold references."""

import pytest
import sacrebleu

from lm_eval.api import metrics


@pytest.mark.parametrize(
    "metric", [metrics.bleu, metrics.chrf, metrics.chrfpp, metrics.ter]
)
@pytest.mark.parametrize("reverse_documents", [False, True])
@pytest.mark.parametrize("reverse_references", [False, True])
def test_translation_metrics_keep_later_correct_reference(
    metric, reverse_documents, reverse_references
):
    refs = [
        ["we have a good example"],
        ["not this reference at all", "this is the correct translation"],
    ]
    preds = ["we have a good example", "this is the correct translation"]
    if reverse_references:
        refs[1].reverse()
    items = list(zip(refs, preds, strict=True))
    if reverse_documents:
        items.reverse()

    expected = 0.0 if metric is metrics.ter else 100.0

    assert metric(items) == pytest.approx(expected)


def test_sacreformat_pads_missing_references_without_mutation():
    refs = [["first"], ["second", "third", "fourth"]]
    preds = ["first", "third"]

    formatted_refs, formatted_preds = metrics._sacreformat(refs, preds)

    assert formatted_refs == [("first", "second"), (None, "third"), (None, "fourth")]
    assert formatted_preds is preds
    assert refs == [["first"], ["second", "third", "fourth"]]


@pytest.mark.parametrize("multireference", [False, True])
def test_sacreformat_uniform_reference_counts_preserve_scores(multireference):
    refs = [
        ["we have a good example", "another good reference example"],
        ["this is a different translation", "this reference is equally valid"],
    ]
    if not multireference:
        refs = [row[0] for row in refs]
    preds = ["we have a small example", "this is a separate translation"]
    expected_refs = list(zip(*refs, strict=True)) if multireference else [tuple(refs)]
    items = list(zip(refs, preds, strict=True))

    assert metrics.bleu(items) == pytest.approx(
        sacrebleu.corpus_bleu(preds, expected_refs).score
    )
    assert metrics.chrf(items) == pytest.approx(
        sacrebleu.corpus_chrf(preds, expected_refs).score
    )
    assert metrics.chrfpp(items) == pytest.approx(
        sacrebleu.corpus_chrf(preds, expected_refs, word_order=2).score
    )
    assert metrics.ter(items) == pytest.approx(
        sacrebleu.corpus_ter(preds, expected_refs).score
    )
