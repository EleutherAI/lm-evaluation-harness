import json
import string

import pytest
from datasets import Dataset

from lm_eval.tasks.score.utils import (
    option_order_robustness_process_docs,
    options_consistency_rate,
)


def _results(tmp_path, num_labels, num_options, pick):
    labels = list(string.ascii_uppercase[:num_labels])
    template = tmp_path / "templates.json"
    template.write_text(
        json.dumps({"key": {"prompt": "", "options_format": ""}}), encoding="utf-8"
    )
    docs = Dataset.from_dict(
        {
            "question_id": list(range(num_options)),
            "options": [[f"o{i}" for i in range(num_options)]] * num_options,
            "answer_index": list(range(num_options)),
            "answer": labels[:num_options],
        }
    )
    copies = option_order_robustness_process_docs(docs, str(template), "key", labels)
    return [
        (
            doc["question_id"],
            doc["always_same_option"],
            pick(doc, labels),
            doc["original_answer_index"],
            (doc["answer_index"],),
        )
        for doc in copies
    ], labels


def _perfect(doc, labels):
    return labels[doc["answer_index"]]


def _same_wrong_option(doc, labels):
    gold = doc["original_answer_index"]
    wrong = f"o{(gold + 1) % len(doc['options'])}"
    return labels[doc["options"].index(wrong)]


def _original_gold_letter(doc, labels):
    return labels[doc["original_answer_index"]]


# Questions with fewer options than labels get copies that keep the original
# order, which is the case a plain swap gets wrong.
@pytest.mark.parametrize(
    "num_labels,num_options", [(5, 5), (5, 4), (10, 10), (10, 8), (10, 4)]
)
@pytest.mark.parametrize("pick", [_perfect, _same_wrong_option])
def test_consistent_models_score_one(tmp_path, num_labels, num_options, pick):
    results, labels = _results(tmp_path, num_labels, num_options, pick)
    assert options_consistency_rate(results, labels) == pytest.approx(1.0)


@pytest.mark.parametrize("num_labels", [5, 10])
def test_position_bias_scores_zero_on_full_length_questions(tmp_path, num_labels):
    results, labels = _results(tmp_path, num_labels, num_labels, _original_gold_letter)
    assert options_consistency_rate(results, labels) == pytest.approx(0.0)
