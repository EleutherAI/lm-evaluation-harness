import pytest

from lm_eval.tasks.ifeval.instructions import BulletListChecker


def _make_checker(num_bullets: int) -> BulletListChecker:
    checker = BulletListChecker("detectable_format:number_bullet_lists")
    checker.build_description(num_bullets=num_bullets)
    return checker


@pytest.mark.parametrize(
    "response,expected_count",
    [
        # 1. Standard valid bullets (* and -)
        ("* First item\n* Second item\n* Third item", 3),
        ("- First item\n- Second item", 2),
        ("* Asterisk item\n- Hyphen item", 2),
        # 2. Indented / nested bullets (spaces and tabs)
        ("  * Indented with spaces\n\t- Indented with tab\n    * Deeply indented", 3),
        # 3. Multiple spaces or tabs after the bullet marker
        ("*   Three spaces after asterisk\n-\tTab after hyphen", 2),
        # 4. Single-character bullet content
        ("* A\n- B\n* 1", 3),
        # 5. Valid bullets starting with inline formatting or negative numbers
        (
            (
                "* **Bold title:** description here\n"
                "* *Italic title:* description here\n"
                "* ***Bold-italic title:*** description here\n"
                "- **Bold on hyphen bullet**\n"
                "- -5 degrees Celsius is freezing\n"
                "* -3.14 is negative pi"
            ),
            6,
        ),
        # 6. Windows CRLF (\r\n) line endings
        ("* First CRLF bullet\r\n- Second CRLF bullet\r\n---\r\n*Italic*\r\n", 2),
    ],
)
def test_bullet_list_checker_valid_cases(response: str, expected_count: int):
    """Verify valid markdown bullet variations are accurately counted."""
    checker = _make_checker(expected_count)
    assert checker.check_following(response) is True

    # Also verify exact count enforcement (off-by-one should return False)
    wrong_checker = _make_checker(expected_count + 1)
    assert wrong_checker.check_following(response) is False


@pytest.mark.parametrize(
    "false_positive_line",
    [
        # Markdown horizontal rules (unspaced, spaced, indented)
        "---",
        "----",
        "***",
        "****",
        "- - -",
        "* * *",
        "-  -  -",
        "*  *  *",
        "   ---   ",
        "   * * *   ",
        # Negative numbers & mathematical expressions at line start
        "-5",
        "-3.14 is a negative float",
        "-100% reduction",
        "-.5 is a negative decimal",
        # Markdown italics, bold, and bold-italic lines
        "*This entire line is italicized*",
        "*Italic start* followed by normal text",
        "**Bold line at the start of a paragraph**",
        "***Bold and italic line***",
        "  *Indented italic line*",
        # Double hyphens, em-dash style attributions, and CLI flags
        "--verbose flag is enabled",
        "-- Author Name",
        "--- Quote attribution",
        # Empty bullet markers with no text content
        "*",
        "-",
        "*   ",
        "-   ",
        " \t * \t ",
    ],
)
def test_bullet_list_checker_rejects_false_positives(false_positive_line: str):
    """Ensure non-bullet constructs never increment the bullet count (#4507)."""
    combined_response = f"{false_positive_line}\n* Legitimate bullet point\n"
    checker_one = _make_checker(1)
    assert checker_one.check_following(combined_response) is True

    # Confirm it was not counted as 2 bullets
    checker_two = _make_checker(2)
    assert checker_two.check_following(combined_response) is False


def test_bullet_list_checker_no_multiline_bleed():
    r"""Ensure a trailing bare '*' or '-' on one line does not bleed across '\n'."""
    response = (
        "Here is a dangling asterisk:\n"
        "*\n"
        "Normal paragraph line right after the dangling asterisk.\n"
        "-\n"
        "Normal paragraph line right after the dangling hyphen.\n"
        "* Actual bullet 1\n"
        "- Actual bullet 2\n"
    )
    checker = _make_checker(2)
    assert checker.check_following(response) is True


def test_bullet_list_checker_realistic_llm_response():
    """End-to-end test simulating a typical formatted LLM output."""
    llm_output = (
        "Here is the breakdown of the temperature data:\n"
        "---\n"
        "*Note: All measurements are in degrees Celsius.*\n\n"
        "* **Day 1:** Recorded a low of -5 degrees.\n"
        "- **Day 2:** Dropped further to -12 degrees.\n"
        "* *Day 3:* Recovered slightly.\n\n"
        "-5 degrees was the median temperature across the period.\n"
        "***\n"
        "* * *\n"
        "- - -\n"
        "**Conclusion:** The cold wave has passed."
    )
    checker_three = _make_checker(3)
    assert checker_three.check_following(llm_output) is True
