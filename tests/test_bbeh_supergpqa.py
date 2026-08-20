import pytest

from lm_eval.tasks import TaskManager
from lm_eval.tasks.bbeh import utils as bbeh_utils
from lm_eval.tasks.supergpqa import utils as supergpqa_utils


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("Ok The final answer is: \\boxed{4}.", "4", True),
        ("[Reasoning] The final answer is: \\boxed{4}.", "3", False),
        ("Alright! The final answer is: 2, 3, 4", "2,3,4", True),
        ("Ok The answer is: (A)", "a", True),
        ("Ok The answer is: **25**\nHere's why.", "25.0", True),
        ("The answer is: [yes]", "yes", True),
        ("The answer is: can't", "cant", True),
        ("The answer is: yes?", "yes", True),
    ],
)
def test_bbeh_official_evaluator(prediction, reference, expected):
    assert bbeh_utils.evaluate_correctness(prediction, reference) is expected


def test_bbeh_official_prompt_suffix():
    prompt = bbeh_utils.doc_to_text({"input": "Question?\n"})
    assert prompt.startswith("Question?\n\nThink step by step")
    assert prompt.endswith('For example, "The answer is: (a)".')


def test_bbeh_harmonic_mean():
    expected = 2 / (1 / 0.51 + 1 / 0.26)
    assert bbeh_utils.harmonic_mean([0.5, 0.25]) == pytest.approx(expected)
    assert bbeh_utils.harmonic_mean([0.0, 0.0]) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        bbeh_utils.harmonic_mean([])


def test_bbeh_harmonic_mean_reproduces_published_random_baseline():
    published_task_percentages = [
        33.3,
        20.0,
        0.0,
        38.0,
        21.0,
        1.4,
        6.2,
        0.0,
        0.0,
        10.0,
        0.0,
        10.0,
        0.0,
        1.6,
        12.5,
        14.3,
        5.2,
        0.0,
        0.0,
        0.5,
        5.5,
        4.3,
        15.4,
    ]
    score = bbeh_utils.harmonic_mean(
        [percentage / 100 for percentage in published_task_percentages]
    )
    assert score * 100 == pytest.approx(2.4, abs=0.05)


@pytest.fixture
def supergpqa_doc():
    return {
        "question": "Which option is correct?",
        "options": ["Alpha", "Beta", "Gamma"],
        "answer_letter": "B",
        "discipline": "Science",
        "field": "Physics",
        "subfield": "Optics",
        "difficulty": "middle",
    }


def test_supergpqa_official_prompt(supergpqa_doc):
    prompt = supergpqa_utils.doc_to_text(supergpqa_doc)
    assert prompt.endswith("Which option is correct?\nA) Alpha\nB) Beta\nC) Gamma\n")
    assert "Answer: $LETTER" in prompt


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Reasoning\nAnswer: B", "B"),
        ("The best option is $\\boxed{C}:", "C"),
        ("Reasoning\n(B).", "B"),
    ],
)
def test_supergpqa_extracts_option_labels(response, expected):
    assert supergpqa_utils.extract_option_label(response, 3) == expected


def test_supergpqa_falls_back_to_option_content():
    assert (
        supergpqa_utils.extract_answer("Answer: Beta value", ["A", "Beta value"])[0]
        == "B"
    )


def test_supergpqa_metrics_match_sample_and_hierarchy_averages():
    records = [
        (1.0, "D1", "F1", "S1", "easy", "parsed"),
        (0.0, "D1", "F1", "S1", "middle", "miss"),
        (1.0, "D2", "F2", "S2", "hard", "parsed"),
    ]
    assert supergpqa_utils.aggregate_accuracy(records) == pytest.approx(2 / 3)
    assert supergpqa_utils.aggregate_subfield_accuracy(records) == pytest.approx(0.75)
    assert supergpqa_utils.aggregate_field_accuracy(records) == pytest.approx(0.75)
    assert supergpqa_utils.aggregate_discipline_accuracy(records) == pytest.approx(0.75)
    assert supergpqa_utils.aggregate_easy_accuracy(records) == 1.0
    assert supergpqa_utils.aggregate_middle_accuracy(records) == 0.0
    assert supergpqa_utils.aggregate_hard_accuracy(records) == 1.0
    assert supergpqa_utils.aggregate_miss_rate(records) == pytest.approx(1 / 3)
    assert supergpqa_utils.aggregate_error_rate(records) == 0.0


def test_new_tasks_are_registered():
    task_manager = TaskManager(include_defaults=True)
    expected = {
        "bbeh",
        "bbeh_mini",
        "bbeh_boardgame_qa",
        "bbeh_zebra_puzzles",
        "supergpqa",
        "supergpqa_five_shot",
    }
    assert expected <= set(task_manager.task_index)
