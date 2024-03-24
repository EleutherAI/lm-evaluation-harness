import collections
import re
import sys
import unicodedata

from lm_eval.filters.extraction import Filter, RegexFilter


class ExtendedRegexFilter(RegexFilter):
    punct_tbl = dict.fromkeys(
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
    )

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def filter_ignores(self, st):
        if self.regexes_to_ignore is not None:
            for s in self.regexes_to_ignore:
                st = re.sub(s, "", st)

        if self.ignore_case:
            st = st.lower()

        if self.ignore_punctuation:
            # https://stackoverflow.com/a/266162
            st = st.translate(self.punct_tbl)
        return st

    def find_match(self, regex, resp, convert_dict={}):
        match = regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict:
                match = convert_dict[match]
        return match


class MapRegexFilter(ExtendedRegexFilter):
    def __init__(
        self,
        regex_pattern_to_value: dict = {},
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        """
        regex_pattern_to_value: Match the regex pattern and change the result into the value
        group_select: Selects the (group_select)th match from the findall result. We use the whole regex_patterns, concatenated by |
        ignore_case: Lowers the case of response before matching with the given regex
        ignore_punctuation: Remove the punctuation before matching with the given regex
        regexes_to_ignore: Remove these regexes before matching with the given regex
        """
        super().__init__(
            "|".join(list(regex_pattern_to_value.keys())),
            group_select,
            fallback,
            ignore_case,
            ignore_punctuation,
            regexes_to_ignore,
        )
        self.regex_to_value = {
            re.compile(r): v for r, v in regex_pattern_to_value.items()
        }

    def apply(self, resps, docs):
        filtered_resps = []

        for r in resps:
            filtered = []
            for resp in r:
                whole_match_considering_group_select = self.find_match(
                    self.regex, self.filter_ignores(resp)
                )
                if whole_match_considering_group_select:
                    for regex, mapped_value in self.regex_to_value.items():
                        match = self.find_match(
                            regex,
                            self.filter_ignores(whole_match_considering_group_select),
                        )
                        if match:
                            match = mapped_value
                            break
                if not whole_match_considering_group_select or not match:
                    match = self.fallback

                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


class NumberParseRegexFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        filtered_resps = []
        import regex
        from word2number import w2n

        # https://www.reddit.com/r/regex/comments/11a38uk/parsing_numbers_written_out_as_english_words
        english_number_regex = regex.compile(
            "((?:(?:zero|one|two|three|four|five|(?:twen|thir|for|fif|six|seven|nine)(?|teen|ty)|eight(?:|een|y)|ten|eleven|twelve|fourteen|hundred|thousand|(?:m|b|tr)illion)(?:zero|one|two|three|four|five|(?:twen|thir|for|fif|six|seven|nine)(?:|teen|ty)|eight(?|een|y)|ten|eleven|twelve|fourteen|hundred|thousand|(?:m|b|tr)illion|[^\S\r\n]|,|and|&)+)?(?:zero|one|two|three|four|five|(?:twen|thir|for|fif|six|seven|nine)(?|teen|ty)|eight(?|een|y)|ten|eleven|twelve|fourteen|hundred|thousand|(?:m|b|tr)illion))"
        )

        for r in resps:
            filtered = []
            for resp in r:
                match = self.find_match(self.regex, resp)
                if not match:
                    match = self.find_match(english_number_regex, resp.lower())
                    if match:
                        match = str(w2n.word_to_num(match))
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


class WordSortFilter(Filter):
    """ """

    def apply(self, resps, docs):
        filtered_resps = []

        for r, doc in zip(resps, docs):
            words = doc["input"].split("List:")[1].strip().split()
            regex = re.compile("|".join([f"\\b{w}\\b" for w in words]))
            filtered = []
            for resp in r:
                match = regex.findall(resp)
                match.reverse()
                ordered_words = reversed(
                    collections.OrderedDict(zip(match, [None] * len(match)))
                )
                filtered.append(" ".join(ordered_words))
            filtered_resps.append(filtered)

        return filtered_resps


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            multiple_choices_regex = re.compile(r"\([A-Z]\)([^\n^(]*)")
            match = multiple_choices_regex.findall(doc["input"])
            for m in match:
                m = self.filter_ignores(m.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                f":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = self.find_match(self.regex, resp)
                if not match:
                    match = self.find_match(
                        fallback_regex, self.filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = self.find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps
