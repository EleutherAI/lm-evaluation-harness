import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    retval = 0
    answer = extract_answer(results[0])

    if is_equiv(answer, remove_boxed(last_boxed_only_string(doc["solution"]))):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2:
            return True
        if are_base_form_equivalent(ss1, ss2):
            return True
        return are_unordered_list_equivalent(ss1, ss2)
    except Exception:  # noqa: BLE001
        return str1 == str2


def extract_answer(string):
    answer_region = extract_math_region(string)
    boxed_answers = extract_boxed_answers(answer_region)
    if boxed_answers:
        return ", ".join(boxed_answers)

    boxed_answers = extract_boxed_answers(string)
    if boxed_answers:
        return ", ".join(boxed_answers)

    return answer_region


def extract_math_region(string):
    indices = [pos for pos, char in enumerate(string) if char == "$"]
    if len(indices) <= 1:
        return string
    return string[indices[0] + 1 : indices[-1]]


def extract_boxed_answers(string):
    answers = []
    start = 0
    while True:
        boxed = first_boxed_only_string(string[start:])
        if boxed is None:
            return answers
        answers.append(remove_boxed(boxed))
        start += string[start:].find(boxed) + len(boxed)


def first_boxed_only_string(string):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None

    if string.startswith("\\boxed ", idx) or string.startswith("\\fbox ", idx):
        next_dollar = string.find("$", idx)
        if next_dollar < 0:
            return string[idx:]
        return string[idx:next_dollar]

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


BASE_FORM_RE = re.compile(r"^(?P<value>[A-Za-z0-9+-]+)_(?P<base>\d+)$")
INLINE_FRACTION_RE = re.compile(r"(?<![A-Za-z])(-?\d+)/(\d+)(?!\d)")
NEGATIVE_FRAC_RE = re.compile(r"\\frac\{-(?P<num>[^{}]+)\}\{(?P<den>[^{}]+)\}")


def canonicalize_base_form(string):
    match = BASE_FORM_RE.fullmatch(string)
    if match is None:
        return string
    return match.group("value")


def are_base_form_equivalent(str1, str2):
    return canonicalize_base_form(str1) == canonicalize_base_form(str2)


def parse_simple_comma_answers(string):
    parts = string.split(",")
    if len(parts) <= 1:
        return None
    if any(part == "" for part in parts):
        return None
    if any(any(char in part for char in "\\{}()[]") for part in parts):
        return None
    return sorted(canonicalize_base_form(part) for part in parts)


def are_unordered_list_equivalent(str1, str2):
    parts1 = parse_simple_comma_answers(str1)
    parts2 = parse_simple_comma_answers(str2)
    if parts1 is None or parts2 is None:
        return False
    return parts1 == parts2


def fix_inline_numeric_fracs(string):
    return INLINE_FRACTION_RE.sub(r"\\frac{\1}{\2}", string)


def normalize_negative_fracs(string):
    return NEGATIVE_FRAC_RE.sub(r"-\\frac{\g<num>}{\g<den>}", string)


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    if "\\fbox " in s:
        left = "\\fbox "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    if not s.startswith(left):
        left = "\\fbox{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)
    string = fix_inline_numeric_fracs(string)
    string = normalize_negative_fracs(string)

    return string
