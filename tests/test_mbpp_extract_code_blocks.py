from lm_eval.tasks.mbpp.utils import extract_code_blocks


def test_extract_code_blocks_when_only_closing_fence_present():
    # `mbpp_instruct` includes an opening fence in gen_prefix (```python\n),
    # so model output often ends with only the closing fence.
    generation = "def square_perimeter(side):\n    return 4 * side\n```"
    assert extract_code_blocks(generation).lstrip().startswith("def square_perimeter")


def test_extract_code_blocks_from_fenced_block_with_language():
    generation = "```python\ndef square_perimeter(side):\n    return 4 * side\n```"
    assert extract_code_blocks(generation).lstrip().startswith("def square_perimeter")


def test_extract_code_blocks_from_fenced_block_without_language():
    generation = "```\ndef square_perimeter(side):\n    return 4 * side\n```"
    assert extract_code_blocks(generation).lstrip().startswith("def square_perimeter")

