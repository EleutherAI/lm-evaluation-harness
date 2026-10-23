from pathlib import Path

import datasets

from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.drop.utils import process_results
from lm_eval.utils import apply_template


DROP_DIR = Path(__file__).parents[1] / "lm_eval" / "tasks" / "drop"
NO_DATE = {"day": "", "month": "", "year": ""}


def raw_doc(number="", spans=()):
    # the EleutherAI/drop schema
    return {
        "section_id": "s",
        "query_id": "q",
        "passage": "Passage.",
        "question": "Question?",
        "answer": {
            "number": number,
            "date": NO_DATE,
            "spans": list(spans),
            "worker_id": "",
            "hit_id": "",
        },
        "validated_answers": {
            "number": [],
            "date": [],
            "spans": [],
            "worker_id": [],
            "hit_id": [],
        },
    }


def processed_docs(config):
    raw = datasets.Dataset.from_list(
        [raw_doc(number="41"), raw_doc(spans=["Andre Brown", "Isaac Redman"])]
    )
    return list(config["process_docs"](raw))


def test_drop_fixed_fewshot_target_is_the_gold_answer():
    config = load_yaml(DROP_DIR / "drop_fixed.yaml")
    targets = [
        apply_template(config["doc_to_target"], d) for d in processed_docs(config)
    ]

    assert targets == ["41", "Andre Brown, Isaac Redman"]


def test_drop_fixed_single_answer_target_scores_as_correct():
    # multi-span scoring is unchanged: process_results reads one predicted string
    config = load_yaml(DROP_DIR / "drop_fixed.yaml")
    doc = processed_docs(config)[0]
    target = apply_template(config["doc_to_target"], doc)

    assert process_results(doc, [config["target_delimiter"] + target]) == {
        "em": 1.0,
        "f1": 1.0,
    }


def test_drop_fixed_keeps_drop_scoring_and_stops_at_newline():
    original = load_yaml(DROP_DIR / "default.yaml", resolve_func=False)
    fixed = load_yaml(DROP_DIR / "drop_fixed.yaml", resolve_func=False)

    assert fixed["task"] == "drop_fixed"
    assert fixed["generation_kwargs"]["until"] == ["\n"]
    for key in ("dataset_path", "process_docs", "process_results", "metric_list"):
        assert fixed[key] == original[key]


def test_drop_fixed_is_registered():
    assert "drop_fixed" in TaskManager().all_tasks
