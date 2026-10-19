from pathlib import Path

import datasets
import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.filters import build_filter_ensemble
from lm_eval.tasks._yaml_loader import load_yaml


TASK_DIR = Path(__file__).parents[1] / "lm_eval" / "tasks" / "mmlu_pro"


@pytest.fixture
def filter_pipeline():
    config = load_yaml(TASK_DIR / "_default_template_yaml")
    pipeline = config["filter_list"][0]
    return build_filter_ensemble(
        pipeline["name"],
        [
            (
                step["function"],
                {key: value for key, value in step.items() if key != "function"},
            )
            for step in pipeline["filter"]
        ],
    )


@pytest.mark.parametrize(
    "response, expected",
    [
        ("The answer is (C).", "C"),
        ("answer is B", "B"),
        ("answer is A)", "A"),
        ("answer is C. answer is B.", "C"),
        ("Answer: J\nThe answer is (C).\nD", "C"),
        # Preserve the existing primary match, including its lack of a word boundary.
        ("The answer is Banana", "B"),
        ("The Answer is C. D", "D"),
        ("Answer: J", "J"),
        ("answer:\tC", "C"),
        ("ANSWER: H", "H"),
        ("Answer: B\nC", "B"),
        # The reference's greedy secondary regex selects the last match on the first matching line.
        ("Answer: A then Answer: B", "B"),
        ("Answer: A\nAnswer: B", "A"),
        ("A or B? Finally (J).", "J"),
        ("A\nB\nC", "C"),
        ("(D)", "D"),
        ("ABC DEF GHI", "[invalid]"),
        ("K L", "[invalid]"),
        ("A1 _B C_ xD Dx", "[invalid]"),
        ("éA Aé", "[invalid]"),
        ("a b c", "[invalid]"),
        ("No choice given.", "[invalid]"),
        ("", "[invalid]"),
        (None, "[invalid]"),
    ],
)
def test_mmlu_pro_configured_filter(filter_pipeline, response, expected):
    instance = Instance(
        request_type="generate_until", doc={}, arguments=(), idx=0, resps=[response]
    )
    filter_pipeline.apply([instance])
    assert instance.filtered_resps["custom-extract"] == expected


def test_mmlu_pro_filter_preserves_document_and_repeat_order(filter_pipeline):
    instances = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(),
            idx=index,
            resps=responses,
        )
        for index, responses in enumerate(
            [["Answer: B", "answer is C"], ["A\nJ", "Answer: D"], ["No choice.", "A"]]
        )
    ]
    filter_pipeline.apply(instances)
    assert [inst.filtered_resps["custom-extract"] for inst in instances] == [
        "B",
        "J",
        "[invalid]",
    ]


@pytest.mark.parametrize(
    "response, expected_score", [("Answer: C", 1.0), ("A\nC", 1.0), ("No choice.", 0.0)]
)
def test_mmlu_pro_task_scores_extracted_answer(monkeypatch, response, expected_score):
    doc = {
        "question": "Which option is correct?",
        "options": ["first", "second", "third"],
        "answer": "C",
        "category": "biology",
        "cot_content": "A: Let's think step by step. The answer is (C).",
    }
    dataset = datasets.Dataset.from_list([doc])

    def download(task, dataset_kwargs=None):
        task.dataset = {"test": dataset, "validation": dataset}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    config = load_yaml(TASK_DIR / "mmlu_pro_biology.yaml")
    task = ConfigurableTask(config=config)
    task._instances = [task.construct_requests(doc, task.doc_to_text(doc))]
    task.instances[0].resps = [response]
    task.apply_filters()

    filtered = task.instances[0].filtered_resps["custom-extract"]
    assert task.process_results(doc, [filtered])["exact_match"] == expected_score
