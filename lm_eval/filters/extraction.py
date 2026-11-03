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

        def find_match(regex, resp, convert_dict=None):
            if not isinstance(resp, str):
                resp = ""
            matches = regex.findall(resp)
            if not matches:
                return ""
            match = matches[self.group_select]
            if isinstance(match, tuple):
                non_empty = [item for item in match if item]
                if not non_empty:
                    return ""
                match = non_empty[0]
            match = match.strip()
            return convert_dict.get(match, match) if convert_dict else match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for pattern in self.regexes_to_ignore:
                    st = re.sub(pattern, "", st)
            if self.ignore_case:
                st = st.lower()
            if self.ignore_punctuation:
                st = st.translate(punct_tbl)
            return st

        def normalize_with_spans(text):
            chars = [(char, i, i + 1) for i, char in enumerate(text)]

            if self.regexes_to_ignore is not None:
                for pattern in self.regexes_to_ignore:
                    current = "".join(char for char, _, _ in chars)
                    removed = set()
                    for found in re.finditer(pattern, current):
                        removed.update(range(found.start(), found.end()))
                    chars = [
                        item for i, item in enumerate(chars) if i not in removed
                    ]

            normalized = []
            for char, start, end in chars:
                value = char.lower() if self.ignore_case else char
                for item in value:
                    if self.ignore_punctuation and unicodedata.category(
                        item
                    ).startswith("P"):
                        continue
                    normalized.append((item, start, end))

            return "".join(item[0] for item in normalized), normalized

        filtered_resps = []

        for responses, doc in zip(resps, docs, strict=True):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"
            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            for choice in doc["choices"]:
                original_choice = choice.strip()
                normalized_choice = filter_ignores(original_choice)
                fallback_regexes.append(re.escape(normalized_choice))
                choice_to_alpha.setdefault(normalized_choice, []).append(
                    (original_choice, f"({next_alpha})")
                )
                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"
                next_alpha = chr(ord(next_alpha) + 1)

            fallback_regex = re.compile(
                "|".join(sorted(fallback_regexes, key=len, reverse=True))
            )
            without_paren_fallback_regex = re.compile(
                rf":[\s]*({'|'.join(without_paren_fallback_regexes)})"
            )

            filtered = []

            for response in responses:
                match = find_match(self.regex, response)

                if not match:
                    normalized_response, spans = normalize_with_spans(
                        response if isinstance(response, str) else ""
                    )
                    occurrences = list(fallback_regex.finditer(normalized_response))

                    if occurrences:
                        selected = occurrences[self.group_select]
                        start, end = selected.span()
                        if start < end:
                            original_start = spans[start][1]
                            original_end = spans[end - 1][2]

                            candidates = choice_to_alpha.get(
                                selected.group().strip(), []
                            )

                            # Include one leading punctuation character only
                            # when it matches a candidate choice's first character.
                            if original_start > 0:
                                preceding = response[original_start - 1]
                                if any(
                                    choice.startswith(preceding)
                                    and unicodedata.category(preceding).startswith("P")
                                    for choice, _ in candidates
                                ):
                                    original_start -= 1

                            original_match = response[
                                original_start:original_end
                            ].strip()

                            exact = [
                                label
                                for choice, label in candidates
                                if choice == original_match
                            ]
                            insensitive = [
                                label
                                for choice, label in candidates
                                if choice.casefold() == original_match.casefold()
                            ]

                            if exact:
                                match = exact[0]
                            elif insensitive:
                                match = insensitive[0]

                    if not match:
                        match = find_match(
                            without_paren_fallback_regex,
                            response,
                            without_paren_to_target,
                        )

                if not match:
                    match = self.fallback

                filtered.append(match)

            filtered_resps.append(filtered)

        return filtered_resps
