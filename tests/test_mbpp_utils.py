from lm_eval.tasks.mbpp.utils import extract_code_blocks


def test_extract_code_blocks_preserves_leading_def_for_prompt_opened_fence():
    response = "def square_perimeter(side):\n    return 4 * side\n```"

    assert extract_code_blocks(response) == (
        "def square_perimeter(side):\n    return 4 * side"
    )


def test_extract_code_blocks_accepts_prompt_opened_fence_without_closing_ticks():
    response = "def square_perimeter(side):\n    return 4 * side"

    assert extract_code_blocks(response) == (
        "def square_perimeter(side):\n    return 4 * side"
    )


def test_extract_code_blocks_prefers_explicit_fenced_block_when_present():
    response = (
        "Here is the implementation:\n"
        "```python\n"
        "def square_perimeter(side):\n"
        "    return 4 * side\n"
        "```"
    )

    assert extract_code_blocks(response) == (
        "def square_perimeter(side):\n    return 4 * side"
    )
