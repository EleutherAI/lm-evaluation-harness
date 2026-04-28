import logging
import signal
import warnings
from typing import Dict, List

import datasets


eval_logger = logging.getLogger(__name__)


try:
    import sympy
    from sympy.parsing.latex import parse_latex

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    from math_verify import parse, verify

    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

_SYMPY_WARNING_ISSUED = False


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
    # First check for multiple \boxed{} (e.g. \boxed{3}, \boxed{5}, \boxed{7})
    all_boxed = find_all_boxed_strings(response)
    # Deduplicate while preserving order (models often repeat the final answer)
    seen = set()
    unique_boxed = []
    for b in all_boxed:
        if b not in seen:
            seen.add(b)
            unique_boxed.append(b)
    if len(unique_boxed) > 1:
        try:
            answer = ", ".join(remove_boxed(b) for b in unique_boxed)
        except AssertionError:
            answer = None
    elif len(unique_boxed) == 1:
        try:
            answer = remove_boxed(all_boxed[0])
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

    out = {
        "exact_match": retval,
    }

    # math_verify provides robust LaTeX-aware answer verification
    if HAS_MATH_VERIFY:
        mv_result = verify(
            gold=parse(doc["solution"]),
            target=parse(response),
        )
        out["math_verify"] = 1 if mv_result else 0

    return out


class _timeout:
    """Timeout context manager using SIGALRM (Unix only)."""

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def _sympy_equiv(ss1: str, ss2: str) -> bool:
    """Check symbolic equivalence using SymPy. Returns True/False."""
    try:
        with _timeout(seconds=5):
            try:
                parsed_1 = parse_latex(ss1)
                parsed_2 = parse_latex(ss2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                return False

            try:
                diff = parsed_1 - parsed_2
            except TypeError:
                return False

            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                return False
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {ss1} and {ss2}")
        return False
    except Exception as e:
        eval_logger.debug(f"Failed comparing {ss1} and {ss2} with {e}")
        return False


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    global _SYMPY_WARNING_ISSUED

    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    # Normalize strings; fall back to raw strings if strip_string fails
    ss1 = str1
    ss2 = str2
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
    except Exception:
        pass
    if verbose:
        print(ss1, ss2)
    if ss1 == ss2:
        return True

    # Fall back to SymPy symbolic equivalence if available
    if HAS_SYMPY:
        if _sympy_equiv(ss1, ss2):
            return True
    elif not _SYMPY_WARNING_ISSUED:
        warnings.warn(
            "sympy not installed — string-only equivalence used for hendrycks_math. "
            "Install via `pip install lm-eval[math]` for improved symbolic matching.",
            stacklevel=2,
        )
        _SYMPY_WARNING_ISSUED = True

    return False


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

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


def find_all_boxed_strings(string):
    """Find all \\boxed{} occurrences in a string, handling nested braces."""
    results = []
    search_start = 0
    while True:
        idx = string.find("\\boxed", search_start)
        if idx < 0:
            break
        # Skip if this is \boxed (space-separated, not brace)
        after = idx + len("\\boxed")
        if after >= len(string) or string[after] != "{":
            search_start = after
            continue
        # Find matching closing brace
        i = after
        num_left_braces_open = 0
        right_brace_idx = None
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is not None:
            results.append(string[idx : right_brace_idx + 1])
            search_start = right_brace_idx + 1
        else:
            search_start = after
    return results


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
