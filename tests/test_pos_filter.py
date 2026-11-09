import pytest

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.instance import Instance
from lm_eval.filters.extraction import POSFilter


@pytest.mark.parametrize(
    "response",
    [
        "[('dog', 'NOUN'), ('runs', 'VERB')]",
        '[("dog", "NOUN"), ("runs", "VERB")]',
        "[('dog',\"NOUN\"), (\"runs\",'VERB')]",
        "[('dog','NOUN'),('runs','VERB')]",
        "[( 'dog' , 'NOUN' ), ( 'runs' , 'VERB' )]",
        "[('dog',\n'NOUN'),\n('runs',\t'VERB')]",
        "[(\"l'école\", 'NOUN'), ('runs', 'VERB')]",
        r"[('d\'og', 'NOUN'), ('runs', 'VERB')]",
        r'[("d\"og", "NOUN"), ("runs", "VERB")]',
        "[('dog', 'NOUN',), ('runs', 'VERB',)]",
    ],
)
def test_pos_filter_accepts_quoted_tuple_formats(response):
    assert list(POSFilter().apply([[response]], [{}])) == [[["NOUN", "VERB"]]]


@pytest.mark.parametrize("delimiter", ["'", '"'], ids=["single", "double"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_pos_filter_accepts_escaped_newlines_in_tokens(delimiter, newline):
    response = f"[({delimiter}do\\{newline}g{delimiter}, 'NOUN'), ('runs', 'VERB')]"

    assert list(POSFilter().apply([[response]], [{}])) == [[["NOUN", "VERB"]]]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Here are the tags: [('dog', 'NOUN'), ('runs', 'VERB')].", ["NOUN", "VERB"]),
        ([("dog", "NOUN"), ("runs", "VERB")], ["NOUN", "VERB"]),
        ("[]", ["invalid"]),
        ("No tags available", ["invalid"]),
        ("[['dog', 'NOUN']]", ["invalid"]),
        ("[('dog', 'NOUN', 'extra')]", ["invalid"]),
    ],
)
def test_pos_filter_preserves_existing_response_behavior(response, expected):
    assert list(POSFilter().apply([[response]], [{}])) == [[expected]]


def test_pos_filter_keeps_custom_fallback():
    assert list(POSFilter(fallback=["missing"]).apply([["[]"]], [{}])) == [
        [["missing"]]
    ]


def test_pos_filter_preserves_document_and_repeat_order_in_pipeline():
    instances = [
        Instance(
            request_type="generate_until",
            doc={"tokens": ["l'école", "opens"]},
            arguments=("prompt", {}),
            idx=0,
            resps=[
                "[(\"l'école\", 'NOUN'), ('opens', 'VERB')]",
                '[("l’école", "PROPN"), ("opens", "VERB")]',
            ],
        ),
        Instance(
            request_type="generate_until",
            doc={"tokens": ["a", "dog"]},
            arguments=("prompt", {}),
            idx=0,
            resps=["[('a','DET'),('dog','NOUN')]", "[]"],
        ),
    ]
    FilterEnsemble(name="pos", filters=[POSFilter]).apply(instances)

    assert instances[0].filtered_resps["pos"] == [
        ["NOUN", "VERB"],
        ["PROPN", "VERB"],
    ]
    assert instances[1].filtered_resps["pos"] == [["DET", "NOUN"], ["invalid"]]
