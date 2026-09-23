"""CPU regressions for the canonical IFEval language checkers."""

import pytest


langdetect = pytest.importorskip("langdetect")
pytest.importorskip("immutabledict")
pytest.importorskip("nltk", minversion="3.9.1")

from lm_eval.tasks.ifeval import utils


@pytest.mark.parametrize(
    "instruction_id,kwargs,response",
    [
        ("language:response_language", {"language": "en"}, "Hello world"),
        ("change_case:english_capital", {}, "THIS IS A TEST"),
        ("change_case:english_lowercase", {}, "we are here"),
    ],
)
def test_language_scores_are_repeatable(monkeypatch, instruction_id, kwargs, response):
    # Other tasks can configure langdetect's global factory independently.
    monkeypatch.setattr(langdetect.DetectorFactory, "seed", 42)
    external_language = langdetect.detect(response)
    doc = {
        "key": 0,
        "prompt": "",
        "instruction_id_list": [instruction_id],
        "kwargs": [kwargs],
    }
    expected = {
        "prompt_level_strict_acc": True,
        "inst_level_strict_acc": [True],
        "prompt_level_loose_acc": True,
        "inst_level_loose_acc": [True],
    }
    for _ in range(10):
        assert utils.process_results(doc, [response]) == expected
    assert langdetect.detect(response) == external_language


@pytest.mark.parametrize(
    "instruction_id,kwargs,response,expected",
    [
        ("language:response_language", {"language": "fr"}, "Hello world", False),
        ("language:response_language", {"language": "en"}, "123", True),
        ("language:response_language", {"language": "en"}, "", False),
    ],
)
def test_language_scoring_preserves_edge_cases(
    instruction_id, kwargs, response, expected
):
    doc = {
        "key": 0,
        "prompt": "",
        "instruction_id_list": [instruction_id],
        "kwargs": [kwargs],
    }
    result = utils.process_results(doc, [response])
    assert result["inst_level_strict_acc"] == [expected]
    assert result["inst_level_loose_acc"] == [expected]
