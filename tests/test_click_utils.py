from lm_eval.tasks.click.click_cul import utils as cul_utils
from lm_eval.tasks.click.click_lang import utils as lang_utils


def _check(utils):
    four = {"paragraph": "p", "question": "q", "choices": ["a", "b", "c", "d"]}
    five = {"paragraph": "", "question": "q", "choices": ["a", "b", "c", "d", "e"]}
    assert "A:a, B: b, C: c, D: d" in utils.get_context(four)
    assert "A, B, C, D 중에" in utils.get_context(four)
    assert "A:a, B: b, C: c, D: d, E: e" in utils.get_context(five)
    assert "A, B, C, D, E 중에" in utils.get_context(five)


def test_click_context_renders_all_options():
    _check(cul_utils)
    _check(lang_utils)
