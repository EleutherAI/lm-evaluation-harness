import re
from typing import Dict, List

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


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    response = results[0]

    # Try to extract \boxed{} from the response (matches aime task behavior)
    boxed = last_boxed_only_string(response)
    if boxed is not None:
        try:
            answer = remove_boxed(boxed)
        except AssertionError:
            answer = None
    else:
        answer = None

    # Fall back to $...$ extraction
    if answer is None:
        indices = [pos for pos, char in enumerate(response) if char == "$"]
        if len(indices) <= 1:
            answer = response
        else:
            answer = response[indices[0] + 1 : indices[-1]]

    if is_equiv(answer, remove_boxed(last_boxed_only_string(doc["solution"]))):
        retval = 1

    return {
        "exact_match": retval,
        **process_results_flex(doc, results),
    }


def process_results_flex(doc: dict, results: List[str]) -> Dict[str, int]:
    """Improved exact match with answer normalization to reduce false negatives."""
    retval = 0
    response = results[0]

    # Try to extract \boxed{} from the response
    # Work on the last line containing \boxed to avoid picking up intermediate steps
    last_idx = response.rfind("\\boxed")
    if last_idx >= 0:
        line_start = response.rfind("\n", 0, last_idx) + 1
        last_line = response[line_start:]
        all_boxed = all_boxed_strings(last_line)
        if len(all_boxed) > 1:
            # Multiple boxed values on the same line (e.g., \boxed{3}, \boxed{5}, \boxed{7})
            answer = ", ".join(remove_boxed(b) for b in all_boxed)
        else:
            boxed = last_boxed_only_string(last_line)
            try:
                answer = remove_boxed(boxed)
            except (AssertionError, TypeError):
                answer = None
    else:
        answer = None

    # Fall back to $...$ extraction
    if answer is None:
        indices = [pos for pos, char in enumerate(response) if char == "$"]
        if len(indices) <= 1:
            answer = response
        else:
            answer = response[indices[0] + 1 : indices[-1]]

    target = remove_boxed(last_boxed_only_string(doc["solution"]))
    original_target = target

    if answer is not None:
        # Strip ordinal suffix ^{\text{...}} before \text{} stripping (e.g., doc_id=379: 12^{\text{th}} → 12)
        answer = re.sub(r"\^\{\\text\{[^}]*\}\}$", "", answer)
    # Strip \text{...} from both target and answer
    target = re.sub(r"\\text\{([^}]*)\}", r"\1", target)
    if answer is not None:
        answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    # Strip base subscripts from numeric answers (e.g., 4210_{5} → 4210, 2516_8 → 2516)
    # Only strip when the main part is a pure integer, to avoid corrupting symbolic answers like x_2
    target = re.sub(r"^(\d+)_\{?\d+\}?$", r"\1", target)
    if answer is not None:
        answer = re.sub(r"^(\d+)_\{?\d+\}?$", r"\1", answer)

    # Build accepted_targets: the original target plus normalized variants
    accepted_targets = [target]

    # Use target \text{...} with `\text` content fully removed as an accepted variant
    # e.g., doc_id=301: "5.4 \text{ cents}" → "5.4"
    target_text_stripped = re.sub(r"\\text\{[^}]*\}", "", original_target).strip()
    if target_text_stripped and target_text_stripped != target:
        accepted_targets.append(target_text_stripped)

    # doc_id=97: lowercase (e.g., "East" → "east")
    if target.lower() != target:
        accepted_targets.append(target.lower())

    # doc_id=227/255/296: single-letter multiple choice — accept with or without surrounding parens
    # e.g., "(C)" accepted as "C", or "C" accepted as "(C)"
    if re.match(r"^\([A-Za-z]\)$", target.strip()):
        accepted_targets.append(target.strip()[1:-1])  # strip parens: (C) → C
    elif re.match(r"^[A-Za-z]$", target.strip()):
        accepted_targets.append(f"({target.strip()})")  # add parens: C → (C)

    # doc_id=343/198/217: thousands separators and LaTeX thin-spaces (\!) in numeric targets
    # Normalize to plain integer and also accept thousands-comma form, so both directions work:
    # e.g., target "58,500" → also accept "58500"; target "7452714" → also accept "7,452,714"
    # doc_id=217: "11,\! 111,\! 111,\! 100" (spaces after \!) → also accept "11111111100"
    t_numeric = re.sub(r"[,\s\\!]", "", target)
    if re.match(r"^\d+$", t_numeric):
        t_formatted = f"{int(t_numeric):,}"
        if t_numeric != target:
            accepted_targets.append(t_numeric)        # plain: 58500
        if t_formatted != target:
            accepted_targets.append(t_formatted)      # formatted: 58,500

    # doc_id=383: "z \in [-2,7]" — also accept just the interval, for any single-letter variable
    t_interval = re.sub(r"^[A-Za-z]\s*\\in\s*", "", target.strip())
    if t_interval != target.strip():
        accepted_targets.append(t_interval)

    # doc_id=467/257: \mbox{ cm}^2 or \mbox{ inches}^2 units — also accept bare number
    # e.g., "864 \mbox{ inches}^2" → "864", "15\mbox{ cm}^2" → "15"
    t_no_units = re.sub(r"\s*\\mbox\{[^}]*\}(\^\{?[^}\s]*\}?)?", "", target).strip()
    if t_no_units != target:
        accepted_targets.append(t_no_units)

    # Build candidate_answers: the extracted answer plus normalized variants
    candidate_answers = [answer] if answer is not None else []
    # doc_id=97: lowercase (e.g., answer "East" → also try "east")
    if answer is not None and answer.lower() != answer:
        candidate_answers.append(answer.lower())

    # doc_id=25/36: for bare comma-separated lists, add sorted form to both accepted_targets and candidate_answers
    # so order doesn't matter (e.g., "-2, 1" vs "1,-2").
    # Excludes ordered tuples/pairs wrapped in parens/brackets where order matters,
    # and excludes pure-integer targets whose commas are thousands separators (e.g., "58,500").
    if "," in target and not re.search(r"^[\(\[\{\\]", target.strip()) and not re.match(r"^\d+$", t_numeric):
        sorted_target = ", ".join(sorted(p.strip() for p in target.split(",")))
        if sorted_target != target:
            accepted_targets.append(sorted_target)
        if answer is not None and "," in answer and not re.search(r"^[\(\[\{\\]", answer.strip()):
            sorted_answer = ", ".join(sorted(p.strip() for p in answer.split(",")))
            if sorted_answer != answer:
                candidate_answers.append(sorted_answer)

    if any(is_equiv(a, t) for a in candidate_answers for t in accepted_targets):
        retval = 1

    return {
        "flexible_match": retval,
    }


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
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def all_boxed_strings(string):
    """Return all \\boxed{...} substrings in order, with proper brace matching."""
    results = []
    idx = 0
    while True:
        start = string.find("\\boxed{", idx)
        if start < 0:
            break
        depth = 0
        for j in range(start, len(string)):
            if string[j] == "{":
                depth += 1
            elif string[j] == "}":
                depth -= 1
                if depth == 0:
                    results.append(string[start : j + 1])
                    idx = j + 1
                    break
        else:
            break
    return results


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
        assert string == "{}/{}".format(a, b)
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

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
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

    return string
