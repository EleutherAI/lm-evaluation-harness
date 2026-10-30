"""Synthetic CoQA regressions; no benchmark passages or model downloads."""

from copy import deepcopy
from itertools import permutations
from unittest.mock import patch

import datasets
import pytest

from lm_eval.tasks import TaskManager
from lm_eval.tasks.coqa import utils


def make_doc(answers):
    return {
        "story": "An invented story used only to test the evaluation pipeline.",
        "questions": {"input_text": ["Who arrived?", "What was the color?"]},
        "answers": {"input_text": ["HISTORY_SENTINEL", answers[0]]},
        "additional_answers": {
            str(i): {"input_text": ["OTHER_HISTORY", answer]}
            for i, answer in enumerate(answers[1:])
        },
    }


@pytest.mark.parametrize(
    "answers,prediction,expected",
    [
        (["red", "red", "blue", "blue"], "red", {"em": 1.0, "f1": 1.0}),
        (["red", "red", "red", "blue"], "blue", {"em": 0.75, "f1": 0.75}),
        (["RED", "red", "blue", "blue"], "red", {"em": 1.0, "f1": 1.0}),
        (
            ["red fox", "red fox", "blue whale", "blue whale"],
            "fox",
            {"em": 0.0, "f1": 2 / 3},
        ),
        (["red", "blue", "green", "yellow"], "red", {"em": 0.75, "f1": 0.75}),
        (["red"] * 4, "red", {"em": 1.0, "f1": 1.0}),
        (["red"], "red", {"em": 1.0, "f1": 1.0}),
        (["red", "red", "blue", "blue"], "green", {"em": 0.0, "f1": 0.0}),
        (["", "", "red", "red"], "", {"em": 1.0, "f1": 1.0}),
    ],
)
def test_annotator_weighted_scores(answers, prediction, expected):
    doc = make_doc(answers)
    assert utils.doc_to_target(doc) == answers
    assert utils.process_results(doc, [prediction]) == pytest.approx(expected)
    assert utils.em(utils.doc_to_target(doc), prediction) == expected["em"]


def test_annotator_order_does_not_affect_score():
    for answers in set(permutations(["red", "red", "red", "blue"])):
        doc = make_doc(answers)
        assert utils.process_results(doc, ["blue"]) == {"em": 0.75, "f1": 0.75}


def test_target_annotations_do_not_leak_into_prompt():
    doc = make_doc(["TARGET_SENTINEL", "ALT_SENTINEL", "ALT_SENTINEL"])
    before = deepcopy(doc)
    prompt = utils.doc_to_text(doc)
    assert "HISTORY_SENTINEL" in prompt
    assert "OTHER_HISTORY" not in prompt
    assert "TARGET_SENTINEL" not in prompt
    assert "ALT_SENTINEL" not in prompt
    assert prompt.endswith("Q: What was the color?\n\nA:")
    changed = make_doc(["CHANGED_TARGET", "CHANGED_ALT"])
    assert utils.doc_to_text(changed) == prompt
    assert doc == before


@pytest.mark.parametrize("additional", [None, {}])
def test_single_annotation_without_additional_answers(additional):
    doc = make_doc(["red"])
    if additional is None:
        doc.pop("additional_answers")
    else:
        doc["additional_answers"] = additional
    assert utils.doc_to_target(doc) == ["red"]
    assert utils.process_results(doc, [" red\nQ: ignored"]) == {"em": 1.0, "f1": 1.0}


def test_registered_task_loads_and_builds_answer_free_requests():
    """Exercise real YAML resolution, dataset conversion, prompt and scoring hooks."""
    doc = make_doc(["TARGET_SENTINEL"] * 2 + ["ALT_SENTINEL"] * 2)
    dataset = datasets.DatasetDict(
        {split: datasets.Dataset.from_list([doc]) for split in ("train", "validation")}
    )
    with patch("datasets.load_dataset", return_value=dataset) as load:
        manager = TaskManager()
        assert "coqa" in manager.all_tasks
        task = manager.load("coqa")["tasks"]["coqa"]
    assert load.call_args.kwargs["path"] == "EleutherAI/coqa"
    assert task.config.validation_split == "validation"
    loaded = task.validation_docs()[0]
    assert (
        task.doc_to_target(loaded)
        == doc["answers"]["input_text"][-1:] * 2 + ["ALT_SENTINEL"] * 2
    )
    context = task.fewshot_context(loaded, num_fewshot=0)
    assert "TARGET_SENTINEL" not in context
    assert "ALT_SENTINEL" not in context
    requests = task.construct_requests(loaded, context, metadata=("coqa", 0, 1))
    assert requests.request_type == "generate_until"
    assert requests.args[0] == context
    assert task.process_results(loaded, ["TARGET_SENTINEL"]) == {"em": 1.0, "f1": 1.0}
    assert task.aggregation()["em"]([1.0, 0.0]) == 0.5
    assert task.aggregation()["f1"]([1.0, 0.0]) == 0.5


def test_offline_evaluator_with_constant_mock():
    """Execution verification only: the mock is not a trained model baseline."""
    from lm_eval.evaluator import evaluate
    from lm_eval.models.dummy import DummyLM

    class ConstantMock(DummyLM):
        def generate_until(self, requests, disable_tqdm=False):
            for request in requests:
                assert "TARGET_SENTINEL" not in request.args[0]
                assert "ALT_SENTINEL" not in request.args[0]
            return ["TARGET_SENTINEL"] * len(requests)

    docs = [
        make_doc(["TARGET_SENTINEL"] * 2 + ["ALT_SENTINEL"] * 2),
        make_doc(["ALT_SENTINEL"] * 4),
    ]
    data = datasets.DatasetDict(
        {split: datasets.Dataset.from_list(docs) for split in ("train", "validation")}
    )
    with patch("datasets.load_dataset", return_value=data):
        tasks = TaskManager().load("coqa")
    tasks["tasks"]["coqa"].set_config("num_fewshot", 0)
    results = evaluate(ConstantMock(), tasks, bootstrap_iters=0)
    assert results["results"]["coqa"]["em,none"] == 0.5
    assert results["results"]["coqa"]["f1,none"] == 0.5
    assert results["versions"]["coqa"] == 4.0
