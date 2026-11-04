import pytest


pytest.importorskip("langdetect")
pytest.importorskip("immutabledict")
pytest.importorskip("nltk")

from lm_eval.tasks.ifeval import instructions, utils


INSTRUCTION_ID = "length_constraints:nth_paragraph_first_word"


def _checker(num_paragraphs, nth_paragraph, first_word):
    checker = instructions.ParagraphFirstWordCheck(INSTRUCTION_ID)
    checker.build_description(
        num_paragraphs=num_paragraphs,
        nth_paragraph=nth_paragraph,
        first_word=first_word,
    )
    return checker


def test_blank_separators_do_not_shift_the_nth_paragraph():
    leading_blank = "\n\nApple starts the first paragraph.\n\nZebra is the second."
    doubled_separator = "Apple starts the first paragraph.\n\n\n\nZebra is the second."
    plain = "Apple starts the first paragraph.\n\nZebra is the second."

    # A leading blank used to make paragraph 2 the first real paragraph.
    assert _checker(2, 2, "apple").check_following(leading_blank) is False
    assert _checker(2, 2, "zebra").check_following(leading_blank) is True
    assert _checker(2, 1, "apple").check_following(leading_blank) is True

    # "\n\n\n\n" is one separator plus an empty segment, not an extra paragraph.
    assert _checker(2, 2, "zebra").check_following(doubled_separator) is True
    assert _checker(2, 2, "apple").check_following(doubled_separator) is False

    # Answers that already have no blank segments stay on the same paragraph.
    assert _checker(2, 2, "zebra").check_following(plain) is True
    assert _checker(2, 2, "apple").check_following(plain) is False
    assert _checker(2, 1, "apple").check_following(plain) is True


def test_scoring_hook_rejects_a_shifted_paragraph():
    doc = {
        "key": 1,
        "prompt": "Write 2 paragraphs. Paragraph 2 must start with apple.",
        "instruction_id_list": [INSTRUCTION_ID],
        "kwargs": [{"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "apple"}],
    }
    response = "\n\nApple starts the first paragraph.\n\nZebra is the second."
    assert utils.process_results(doc, [response]) == {
        "prompt_level_strict_acc": False,
        "inst_level_strict_acc": [False],
        "prompt_level_loose_acc": False,
        "inst_level_loose_acc": [False],
    }
