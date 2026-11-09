from lm_eval.tasks.ifeval import instructions


def test_keyword_checker_rejects_substrings_inside_longer_words():
    checker = instructions.KeywordChecker("kw")
    checker.build_description(keywords=["cat", "art"])
    assert not checker.check_following("The category of the start.")


def test_keyword_checker_accepts_exact_words_and_plurals():
    checker = instructions.KeywordChecker("kw")
    checker.build_description(keywords=["cat", "box"])
    assert checker.check_following("My cats sat on the boxes.")
    assert checker.check_following("A cat and a box.")


def test_keyword_checker_rejects_non_plural_derivations():
    checker = instructions.KeywordChecker("kw")
    checker.build_description(keywords=["disgusting"])
    assert not checker.check_following("He behaved DISGUSTINGLY.")
    assert checker.check_following("That was disgusting.")


def test_keyword_frequency_checker_ignores_substring_matches():
    checker = instructions.KeywordFrequencyChecker("freq")
    checker.build_description(keyword="art", frequency=2, relation="less than")
    assert checker.check_following("I love art. The start.")

    at_least_checker = instructions.KeywordFrequencyChecker("freq_min")
    at_least_checker.build_description(
        keyword="story", frequency=2, relation="at least"
    )
    assert not at_least_checker.check_following("History repeats in history.")
    assert at_least_checker.check_following("One story and another story.")

    plural_freq_checker = instructions.KeywordFrequencyChecker("freq_plural")
    plural_freq_checker.build_description(
        keyword="sneaker", frequency=2, relation="at least"
    )
    assert plural_freq_checker.check_following("One sneaker and two sneakers.")
