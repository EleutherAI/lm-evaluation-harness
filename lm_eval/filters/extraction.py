from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@register_filter("regex")
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """Compile `regex_pattern` and set the fallback for non-matches.

        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> Iterable[list[str]]:
        def filter_set(inst: Sequence[str]) -> list[str]:
            filtered = []
            for resp in inst:
                if not isinstance(resp, str):
                    resp = ""
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m]
                        if match:
                            match = match[0]
                        else:
                            match = self.fallback
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        return [filter_set(x) for x in resps]


@register_filter("regex_pos")
class POSFilter(Filter):
    """Extract part-of-speech tags from model responses."""

    def __init__(
        self,
        regex_pattern: str = r"\['(.*?)'\]",
        group_select=0,
        fallback=None,
    ) -> None:
        """Compile `regex_pattern` and set the fallback for non-matches.

        `fallback` defines the output returned if no matches for the regex are located.
        """
        if fallback is None:
            fallback = ["invalid"]
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]):
        def extract_tagged_tokens(text):
            # Extract tagged tokens list from text input using regex
            tokens = re.findall(r"\('([^']*)', '([^']*)'\)", text)
            return [(token, pos) for token, pos in tokens]

        def extract_pos_tags(result):
            pos_tags = []
            if isinstance(result, str):
                result = extract_tagged_tokens(result)
            pos_tags.extend(pos for _, pos in result)
            return pos_tags or self.fallback

        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = extract_pos_tags(resp)
                filtered.append(match)
            return filtered

        return (filter_set(x) for x in resps)


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """Filters out leading and trailing whitespace from responses."""

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        def filter_set(inst: Sequence[str]) -> list[str]:
            return [resp.strip() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """Extract a model's answer on multiple choice questions with letter answers.

    Assumes each document has a "choices" field containing the list of answer choices
    and that the answer label symbols are of the form (A), (B), (C), ... or A, B, C.

    .. security-note::

        Positional extraction (``group_select`` picks the Nth parenthesized letter
        in the full response) is **position-based, not commitment-based**: a
        response that states ``(A)`` and later writes "Eliminated: (A), (C), (D)"
        will have its extracted letter determined by where letters appear, not by
        which letter the model committed to. This lets a subject model
        adversarially structure its output so the extracted answer differs from
        its intended one, inflating or deflating accuracy without changing
        correctness. For tasks where the model is instructed to emit an explicit
        final-answer marker, prefer :class:`AnchoredMultiChoiceRegexFilter`,
        which extracts the *committed* answer rather than the Nth match.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        regexes_to_ignore: list[str] | None = None,
    ) -> None:
        r"""Configure the multi-choice regex filter.

        Args:
            regex_pattern: The basic regex pattern to use. If it fails to match,
                a customized procedure is used:
                step 1 — parse choices between ([A-Z])s and search in the response.
                step 2 — parse with regex ``r'\s*([A-?])'``, where ``?`` varies by
                number of choices.
            group_select: Selects the (group_select)th match from the findall result.
            ignore_case: Ignore case during step 1 matching.
            ignore_punctuation: Remove punctuation during step 1 matching.
            regexes_to_ignore: Remove these regexes during step 1 matching.
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        import unicodedata

        def find_match(regex, resp, convert_dict: dict[str, str] | None = None):
            if convert_dict is None:
                convert_dict = {}
            if not isinstance(resp, str):
                resp = ""
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    non_empty = [m for m in match if m]
                    if not non_empty:
                        return ""
                    match = non_empty[0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs, strict=True):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


@register_filter("anchored_multi_choice_regex")
class AnchoredMultiChoiceRegexFilter(MultiChoiceRegexFilter):
    """Commitment-based multiple-choice extraction.

    Hardens :class:`MultiChoiceRegexFilter` against position-based answer
    gaming. Instead of trusting the Nth parenthesized letter in the whole
    response (which a subject model can manipulate by where it places letters),
    this filter first looks for an explicit *final-answer sentinel* and extracts
    the letter that immediately follows it. Only if no sentinel is present does
    it fall back to the parent's positional behavior.

    This makes the model's *committed* answer authoritative: a response like

        "Reasoning leads to (B). Eliminated: (A), (C), (D)."

    extracts ``(B)`` (the committed answer) rather than ``(D)`` (the last
    positional match), which is what the parent filter would return.

    Args:
        answer_sentinels: Ordered list of regex patterns that introduce the
            committed answer. The first sentinel that matches is used; the
            captured/matched answer is extracted from the text following it.
            Defaults to common "final answer" phrasings. Set to an empty list
            to disable anchoring and get identical behavior to the parent.
        fallback_to_positional: If ``True`` (default), fall back to the parent
            positional extraction when no sentinel matches. If ``False``,
            responses without a sentinel are scored as ``[invalid]`` — useful
            for tasks that *require* an explicit final answer.
    """

    DEFAULT_SENTINELS: list[str] = [
        # "the/final/correct answer is (X)" / "answer: (X)"
        r"(?:the\s+)?(?:final|correct)?\s*answer\s*(?:is|:)\s*",
        # "therefore, ... (X)" as a common CoT conclusion
        r"therefore[,\s]+(?:the\s+answer\s+is\s*)?",
    ]

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        regexes_to_ignore: list[str] | None = None,
        answer_sentinels: list[str] | None = None,
        fallback_to_positional: bool = True,
    ) -> None:
        super().__init__(
            regex_pattern=regex_pattern,
            group_select=group_select,
            fallback=fallback,
            ignore_case=ignore_case,
            ignore_punctuation=ignore_punctuation,
            regexes_to_ignore=regexes_to_ignore,
        )
        flags = re.IGNORECASE if ignore_case else 0
        sentinels = (
            self.DEFAULT_SENTINELS if answer_sentinels is None else answer_sentinels
        )
        self.answer_sentinels = [re.compile(s, flags) for s in sentinels]
        self.fallback_to_positional = fallback_to_positional

    def _extract_after_sentinel(self, resp: str, letter_regex: re.Pattern) -> str | None:
        """Return the parenthesized letter following the first matching sentinel.

        Returns ``None`` if no sentinel is present or no letter follows it.
        """
        for sentinel in self.answer_sentinels:
            for m in sentinel.finditer(resp):
                tail = resp[m.end():]
                letter_match = letter_regex.search(tail)
                if letter_match:
                    # Normalize bare letter "A" -> "(A)" to match choice_to_alpha form.
                    val = letter_match.group(1)
                    return val if val.startswith("(") else f"({val})"
        return None

    def apply(
        self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
    ) -> list[list[str]]:
        # Letter regex tolerant of both "(A)" and bare "A" forms.
        letter_flags = re.IGNORECASE if self.ignore_case else 0
        letter_regex = re.compile(r"\(?([A-Za-z])\)?", letter_flags)

        # Delegate fallback / non-sentinel responses to the parent's full logic.
        # We only override the response when a sentinel commits to a letter.
        parent_filtered = super().apply(resps, docs)

        hardened = []
        for resp_list, doc, parent_list in zip(resps, docs, parent_filtered, strict=True):
            choices = doc["choices"]
            valid_letters = {
                chr(ord("A") + i): f"({chr(ord('A') + i)})"
                for i in range(len(choices))
            }
            row = []
            for resp, parent_val in zip(resp_list, parent_list, strict=True):
                if not isinstance(resp, str):
                    row.append(parent_val)
                    continue
                anchored = self._extract_after_sentinel(resp, letter_regex)
                if anchored is not None:
                    # Honor ignore_case / canonicalize to the "(X)" form within range.
                    canon = anchored.upper() if self.ignore_case else anchored
                    if canon in valid_letters.values() or canon.strip("()") in valid_letters:
                        letter = canon.strip("()")
                        row.append(valid_letters.get(letter, parent_val))
                    else:
                        # Sentinel present but letter out of range -> trust sentinel signal,
                        # fall to parent (positional) rather than silently inventing an answer.
                        row.append(parent_val)
                elif self.fallback_to_positional:
                    row.append(parent_val)
                else:
                    row.append(self.fallback)
            hardened.append(row)
        return hardened
