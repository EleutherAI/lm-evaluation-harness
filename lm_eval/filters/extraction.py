import re

from lm_eval.api.filter import Filter


class RegexFilter(Filter):
    """


    """

    def __init__(self, regex=r"#### (\-?[0-9\.\,]+)", fallback="[invalid]"):
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex
        self.regex = re.compile(regex)

        self.fallback = fallback

    def apply(self, resps):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair. 
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex.search(resp)
                if match:
                    match = match.group(1).strip()
                    match_str.replace(",", "")
                    # TODO: should we assume any other filtering is performed?
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        # print(resps)
        filtered_resps = list(map(lambda x: filter_set(x), resps))
        # print(filtered_resps)

        return filtered_resps
