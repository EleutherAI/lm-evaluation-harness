import re
import string


REGEX = re.compile(
    "[`*_]*(?i:FINAL ANSWER|Final Answer|Answer|answer is)[`*_]*[:\s]*[`*_]*([A-D])[`*_]*"
)


def flexible_extract(resps, docs):
    def filter_set(inst):
        filtered = []
        for resp in inst:
            # first, we try to match the regex pattern
            if match := REGEX.findall(resp):
                match = match[-1]
                if match:
                    return match
            # if we can't match the regex pattern, we try to match the last character
            while resp[-1] in string.punctuation:
                resp = resp[:-1]
            if resp[-1] in ["A", "B", "C", "D"]:
                resp = resp[-1]
            else:
                # match on A-D after a colon (last match), for example option: A.
                pattern = r":\s*([A-D])"
                matches = re.findall(pattern, resp)
                if matches:
                    resp = matches[-1]
            filtered.append(resp)
        return filtered

    filtered_resps = list(map(lambda x: filter_set(x), resps))

    return filtered_resps
