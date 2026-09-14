import pytest

from lm_eval.tasks.econ_logic_qa.utils import (
    doc_to_text,
    extract_order,
    pairwise_accuracy,
    parse_order,
    process_results,
)


DACB = ["D", "A", "C", "B"]


@pytest.mark.parametrize(
    "generation, expected",
    [
        # The shape the gold answers use, which few-shot examples teach.
        (" D, A, C, B", DACB),
        ("D, A, C, B", DACB),
        ("D,A,C,B", DACB),
        ("D; A; C; B", DACB),
        ("D -> A -> C -> B", DACB),
        ("D → A → C → B", DACB),
        ("D > A > C > B", DACB),
        ("D A C B", DACB),
        ("DACB", DACB),
        ("d, a, c, b", DACB),
        # Chat models wrap the answer in a sentence.
        ("The correct order is D, A, C, B.", DACB),
        ("**Answer:** D, A, C, B", DACB),
        ("Answer: D, A, C, B\nQuestion:", DACB),
        ("first D, next A, then C, and finally B", DACB),
        # A model that reasons before committing: the last marker wins.
        (
            (
                "The sequence could be A, B, C, D at first glance.\n"
                "But B must follow C, so the answer is D, A, C, B"
            ),
            DACB,
        ),
        # Nothing recoverable.
        ("", []),
        ("I do not know.", []),
        ("A, A, B, C", []),
        ("A, B, C", []),
        ("ABBA", []),
    ],
)
def test_extract_order(generation, expected):
    assert extract_order(generation) == expected


def test_extract_order_ignores_enumerated_option_text():
    # An option list restated verbatim is not an ordering: a bare "." must not
    # act as a separator, or every echoed prompt would parse as "A, B, C, D".
    generation = "A. Demand rises. B. Prices rise. C. Supply rises. D. Prices fall."
    assert extract_order(generation) == ["A", "B", "C", "D"]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("D, A, C, B", DACB),
        ("A, B, C, D", ["A", "B", "C", "D"]),
        ("D,A,C,B", DACB),
        ("A, A, B, C", []),
    ],
)
def test_parse_order(answer, expected):
    assert parse_order(answer) == expected


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        (DACB, DACB, 1.0),
        (["A", "B", "C", "D"], ["D", "C", "B", "A"], 0.0),
        # One adjacent transposition gets five of the six pairs right.
        (["B", "A", "C", "D"], ["A", "B", "C", "D"], 5 / 6),
        # Moving the first event to the end misplaces the three pairs it is in.
        (["B", "C", "D", "A"], ["A", "B", "C", "D"], 0.5),
        # An unparsable generation scores zero, not chance.
        ([], DACB, 0.0),
    ],
)
def test_pairwise_accuracy(pred, gold, expected):
    assert pairwise_accuracy(pred, gold) == pytest.approx(expected)


def test_process_results_correct():
    doc = {"Answer": "D, A, C, B"}
    assert process_results(doc, [" D, A, C, B"]) == {
        "exact_match": 1.0,
        "pairwise_accuracy": 1.0,
    }


def test_process_results_unparsable_scores_zero_on_both():
    doc = {"Answer": "D, A, C, B"}
    assert process_results(doc, ["I cannot answer that."]) == {
        "exact_match": 0.0,
        "pairwise_accuracy": 0.0,
    }


def test_process_results_wrong_order_gets_partial_credit():
    doc = {"Answer": "A, B, C, D"}
    scores = process_results(doc, ["B, A, C, D"])
    assert scores["exact_match"] == 0.0
    assert scores["pairwise_accuracy"] == pytest.approx(5 / 6)


def test_doc_to_text():
    doc = {
        "Question": "Arrange the following events.",
        "A": "Demand rises.",
        "B": "Prices rise.",
        "C": "Supply rises.",
        "D": "Prices fall.",
        "Answer": "A, B, C, D",
    }
    assert doc_to_text(doc) == (
        "Question: Arrange the following events.\n"
        "A. Demand rises.\n"
        "B. Prices rise.\n"
        "C. Supply rises.\n"
        "D. Prices fall.\n"
        "Answer:"
    )
