import pytest

from lm_eval.tasks.ifeval.instructions import SectionChecker


@pytest.fixture
def section_checker():
    checker = SectionChecker(instruction_id="test")
    checker.build_description(section_spliter="Section", num_sections=2)
    return checker


def test_section_checker_ignores_prose_references(section_checker):
    response = "See Section 1 for background and Section 2 for details."
    assert not section_checker.check_following(response)


@pytest.mark.parametrize(
    "response",
    [
        "Section 1\nFirst section.\nSection 2\nSecond section.",
        "## Section 1\nFirst section.\n## Section 2\nSecond section.",
        "**Section 1**\nFirst section.\n**Section 2**\nSecond section.",
        "> Section 1\nFirst section.\n> Section 2\nSecond section.",
    ],
)
def test_section_checker_counts_line_start_headings(section_checker, response):
    assert section_checker.check_following(response)


def test_section_checker_respects_minimum_section_count():
    checker = SectionChecker(instruction_id="test")
    checker.build_description(section_spliter="Section", num_sections=3)

    response = "Section 1\nFirst.\nSection 2\nSecond."
    assert not checker.check_following(response)
