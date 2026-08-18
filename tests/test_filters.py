from lm_eval.filters.extraction import (
    AnchoredMultiChoiceRegexFilter,
    MultiChoiceRegexFilter,
)


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_choice_text():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"()()",
        ignore_case=True,
        ignore_punctuation=True,
    )

    resps = [["alpha"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(A)"]]


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_bare_letter():
    filt = MultiChoiceRegexFilter(regex_pattern=r"()()")

    resps = [[": B"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(B)"]]


# ---------------------------------------------------------------------------
# Regression tests demonstrating the position-based answer-extraction gaming
# vector in MultiChoiceRegexFilter, and that AnchoredMultiChoiceRegexFilter
# resolves the model's *committed* answer instead of the Nth positional match.
#
# Threat model: a subject model's completion is free-form text. The extractor
# selects a letter by position (group_select). A response that states one
# letter and then summarizes/discusses other letters can be scored as a
# DIFFERENT letter than the one the model committed to. This is an eval-
# integrity issue (silent mis-scoring), not a crash.
# ---------------------------------------------------------------------------

# GPQA / MMLU-flan-cot / GSM8K style config: last-match over "(A)" letters.
_LAST_MATCH = {
    "regex_pattern": r"(\([A-Z]\))",
    "group_select": -1,
    "ignore_case": True,
    "ignore_punctuation": True,
}


def test_parent_last_match_is_gamed_by_trailing_elimination_list():
    """REGRESSION (parent filter): trailing 'Eliminated: (A),(C),(D)' flips the
    extracted letter from the committed (B) to (D). This is the bug the
    anchored filter exists to fix.
    """
    filt = MultiChoiceRegexFilter(**_LAST_MATCH)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["Conclusion: (B). Eliminated: (A), (C), (D)."]]
    # Parent returns (D) — the last positional letter, not the committed (B).
    assert filt.apply(resps, [doc]) == [["(D)"]]


def test_parent_first_match_is_gamed_by_revision():
    """REGRESSION (parent filter, group_select=0): 'First I think (A) ...
    reconsidering: (B)' extracts the abandoned (A).
    """
    filt = MultiChoiceRegexFilter(regex_pattern=r"(\([A-Z]\))", group_select=0)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["First I think (A). But wait, reconsidering: (B) is correct."]]
    assert filt.apply(resps, [doc]) == [["(A)"]]


def test_anchored_extracts_committed_answer_despite_trailing_letters():
    """FIX: with the default 'answer is' sentinel, the committed (B) is
    extracted even when trailing letters would otherwise win positionally.
    """
    filt = AnchoredMultiChoiceRegexFilter(**_LAST_MATCH)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["The answer is (B). Eliminated: (A), (C), (D)."]]
    assert filt.apply(resps, [doc]) == [["(B)"]]


def test_anchored_extracts_committed_answer_with_therefore_sentinel():
    """FIX: 'therefore ... (B)' sentinel is also honored by default."""
    filt = AnchoredMultiChoiceRegexFilter(**_LAST_MATCH)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["Therefore, (B). Note (A) is a common trap, as is (D)."]]
    assert filt.apply(resps, [doc]) == [["(B)"]]


def test_anchored_falls_back_to_positional_when_no_sentinel():
    """BACKWARD COMPAT: when no sentinel is present, anchored behaves like the
    parent (positional extraction) so adopting it never silently breaks runs.
    """
    filt = AnchoredMultiChoiceRegexFilter(**_LAST_MATCH)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    # No 'answer is' / 'therefore' phrasing — pure positional response.
    resps = [["My choice is (B)."]]
    assert filt.apply(resps, [doc]) == [["(B)"]]


def test_anchored_strict_mode_marks_missing_sentinel_invalid():
    """FIX (strict mode): with fallback_to_positional=False, a response with no
    sentinel is scored [invalid] rather than positionally — for tasks that
    *require* an explicit committed answer.
    """
    filt = AnchoredMultiChoiceRegexFilter(**_LAST_MATCH, fallback_to_positional=False)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["My choice is (B)."]]
    assert filt.apply(resps, [doc]) == [["[invalid]"]]


def test_anchored_preserves_baseline_extraction_without_noise():
    """NORMAL CASE: a clean 'The answer is (B)' with no trailing letter noise
    extracts identically under parent and anchored filters.
    """
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["Reasoning. The answer is (B)."]]
    parent = MultiChoiceRegexFilter(**_LAST_MATCH).apply(resps, [doc])
    anchored = AnchoredMultiChoiceRegexFilter(**_LAST_MATCH).apply(resps, [doc])
    assert parent == anchored == [["(B)"]]


def test_anchored_handles_bare_letter_after_sentinel():
    """FIX: sentinel followed by a bare letter 'B' (not '(B)') is normalized to
    '(B)' to match the choice_to_alpha form the scorer expects.
    """
    filt = AnchoredMultiChoiceRegexFilter(**_LAST_MATCH)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["The answer is B. Eliminated: A, C, D."]]
    assert filt.apply(resps, [doc]) == [["(B)"]]


def test_anchored_custom_sentinels():
    """FIX: task authors can supply their own sentinel patterns, e.g. for a
    'Final answer: X' prompt format.
    """
    filt = AnchoredMultiChoiceRegexFilter(
        **_LAST_MATCH, answer_sentinels=[r"final\s+answer\s*:\s*"]
    )
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["Some analysis. Final answer: (B). Discarded: (A), (C), (D)."]]
    assert filt.apply(resps, [doc]) == [["(B)"]]


def test_anchored_ignores_prose_words_after_sentinel():
    """FIX: a prose word following the sentinel must not be mistaken for the
    answer letter. 'The answer is by process of elimination, (A).' must extract
    (A), not (B) (the 'b' in 'by'). This is the word-shadowing regression that
    naive single-letter matching after the sentinel would introduce.
    """
    filt = AnchoredMultiChoiceRegexFilter(**_LAST_MATCH)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    resps = [["The answer is by process of elimination, (A)."]]
    assert filt.apply(resps, [doc]) == [["(A)"]]


def test_anchored_ignores_leading_prose_word_with_common_starts():
    """FIX: multiple common prose openers after the sentinel must not shadow
    the committed answer letter.
    """
    filt = AnchoredMultiChoiceRegexFilter(**_LAST_MATCH)
    doc = {"choices": ["alpha", "beta", "gamma", "delta"]}
    # 'clearly' starts with c, 'after' with a, 'definitely' with d.
    # Both responses share one doc (two requests for the same document).
    resps = [["The answer is clearly (C).", "The answer is after some thought, (A)."]]
    assert filt.apply(resps, [doc]) == [["(C)", "(A)"]]
