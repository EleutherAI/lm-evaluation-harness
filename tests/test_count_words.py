import pytest

from lm_eval.api.task import PerplexityTask, Task

CASES = [
    ("", 0),
    ("   ", 0),
    (" \n\t ", 0),
    ("hello", 1),
    ("hello world", 2),
    ("hello world ", 2),
    (" hello world", 2),
    ("  hello   world\n", 2),
    ("a b c", 3),
]


@pytest.mark.parametrize("doc,expected", CASES)
def test_task_count_words_ignores_empty_tokens(doc, expected):
    assert Task.count_words(doc) == expected


@pytest.mark.parametrize("doc,expected", CASES)
def test_perplexity_task_count_words_ignores_empty_tokens(doc, expected):
    assert PerplexityTask.count_words(doc) == expected
