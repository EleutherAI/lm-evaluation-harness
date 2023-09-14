import os
import pathlib
import re
import collections
import functools
import inspect
import sys
import signal
from typing import List, Callable, TypeVar

T = TypeVar('T')

import sympy
from sympy.parsing.latex import parse_latex


class ExitCodeError(Exception):
    pass


def sh(x):
    if os.system(x):
        raise ExitCodeError()


class timeout:
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

class MajorityVotingMixin:
    """
    Majority voting for an arbitrary definition of equivalence.
    """
    def majority_vote(
            self,
            sampled_answers: List[T],
            correct_answer: T,
            is_equiv : Callable[[T, T], bool] = lambda x, y: x==y
    ):
        """
        Performs majority voting on a list of candidate answers. 
        Returns accuracy and pass rate checked against `correct_answer`.
        Supports arbitrary definitions of equivalence via `is_equiv` argument.
        
        Arguments:
            sampled_answers: List[T], list of sampled answers
            correct_answer: T, ground truth.
            is_equiv: Callable[[T, T], bool], a function that determines when two answers 
                should be treated as equivalent. Default is T-equivalence, i.e `lambda x y: x==y`.
        Returns:
            acc: int, 0/1 for correct/incorrect
            pass_rate: bool, proportion of `sampled_answers` equivalent to `correct_answer`
            votes: List[Tuple[T, int]], for each distinct answer, the amount of votes for that answer. 
                Sorted by descending amount of votes, so that `elected_answer==votes[0][0]`
        """
        if not sampled_answers:
            return 0, 0, []

        answer_votes = {}
        for answer in sampled_answers:
            if answer in answer_votes: 
                answer_votes[answer] += 1
            else:
                counted = False
                for ref in answer_votes:
                    if is_equiv(answer, ref) and not counted:
                        answer_votes[ref] += 1
                        counted=True
                if not counted: 
                    answer_votes[answer] = 1

        votes = list(sorted(answer_votes.items(), key=lambda x: -x[1]))

        elected_answer = votes[0][0]

        if is_equiv(correct_answer, elected_answer):
            acc = 1
            pass_rate = votes[0][1] / len(sampled_answers)
        else:
            acc = 0
            pass_rate = 0
            for candidate, num_votes in answer_votes.items():
                if is_equiv(correct_answer, candidate):
                    pass_rate = num_votes / len(sampled_answers)
                    break

        return acc, pass_rate, votes

class SymbolicMathMixin:
    """
    Methods useful for parsing mathematical expressions from text and determining equivalence of expressions.
    """

    SUBSTITUTIONS = [  # used for text normalize
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
    ]
    REMOVED_EXPRESSIONS = [  # used for text normalizer
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
    ]

    def normalize_tex(self, final_answer: str) -> str:
        """
        Normalizes a string representing a mathematical expression.
        Used as a preprocessing step before parsing methods.

        Copied character for character from appendix D of Lewkowycz et al. (2022)
        """
        final_answer = final_answer.split("=")[-1]

        for before, after in self.SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in self.REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

        # Normalize shorthand TeX:
        #  \fracab -> \frac{a}{b}
        #  \frac{abc}{bef} -> \frac{abc}{bef}
        #  \fracabc -> \frac{a}{b}c
        #  \sqrta -> \sqrt{a}
        #  \sqrtab -> sqrt{a}b
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")

        # Normalize 100,000 -> 100000
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")

        return final_answer

    def parse_tex(self, text: str) -> sympy.Basic:
        """
        Wrapper around `sympy.parse_text` that outputs a SymPy expression.
        Typically, you want to apply `normalize_text` as a preprocessing step.
        """
        try:
            parsed = parse_latex(text)
        except (
            sympy.parsing.latex.errors.LaTeXParsingError,
            sympy.SympifyError,
            TypeError,
        ) as e:
            print(f"failed to parse {text} with exception {e}")
            return None

        return parsed

    def is_exp_equiv(self, x1: sympy.Basic, x2: sympy.Basic, time_limit=5) -> bool:
        """
        Determines whether two sympy expressions are equal.
        """
        try:
            with timeout(seconds=time_limit):
                try:
                    diff = x1 - x2
                except TypeError as e:
                    print(
                        f"Couldn't subtract {x1} and {x2} with exception {e}"
                    )
                    return False

                try:
                    if sympy.simplify(diff) == 0:
                        return True
                    else:
                        return False
                except ValueError as e:
                    print(f"Failed to simplify {x1}-{x2} with {e}")
                    return False
        except TimeoutError as e:
            print(f"Timed out comparing {x1} and {x2}")
            return False

    def is_tex_equiv(self, x1: str, x2: str, time_limit=5) -> bool:
        """
        Determines whether two (ideally normalized using `normalize_text`) TeX expressions are equal.
        """
        return self.is_exp_equiv(self.parse_tex(x1), self.parse_tex(x2), time_limit=time_limit)


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = args_string.split(",")
    args_dict = {}
    for arg in arg_list:
        k, v = arg.split("=")
        args_dict[k] = v
    return args_dict


def join_iters(iters):
    for iter in iters:
        yield from iter


def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []

    if arr:
        yield arr


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def make_disjoint_window(pair):
    """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b


class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [([y[0] for y in x], x[0][1]) for x in arr]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr

    def get_reordered(self):
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res


def positional_deprecated(fn):
    """
    A decorator to nudge users into passing only keyword args (`kwargs`) to the
    wrapped function, `fn`.
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(
                f"WARNING: using {fn.__name__} with positional arguments is "
                "deprecated and will be disallowed in a future version of "
                "lm-evaluation-harness!"
            )
        return fn(*args, **kwargs)

    return _wrapper


@positional_deprecated
def find_test_root(start_path: pathlib.Path) -> pathlib.Path:
    """
    Search upward in the directory tree to a maximum of three layers
    to find and return the package root (containing the 'tests' folder)
    """
    cur_path = start_path.resolve()
    max_layers = 3
    for _ in range(max_layers):
        if (cur_path / "tests" / "test_version_stable.py").exists():
            return cur_path
        else:
            cur_path = cur_path.parent.resolve()
    raise FileNotFoundError(
        f"Unable to find package root within {max_layers} upwards" + f"of {start_path}"
    )


@positional_deprecated
def run_task_tests(task_list: List[str]):
    """
    Find the package root and run the tests for the given tasks
    """
    import pytest

    package_root = find_test_root(start_path=pathlib.Path(__file__))
    task_string = " or ".join(task_list)
    args = [
        f"{package_root}/tests/test_version_stable.py",
        f"--rootdir={package_root}",
        "-k",
        f"{task_string}",
    ]
    sys.path.append(str(package_root))
    pytest_return_val = pytest.main(args)
    if pytest_return_val:
        raise ValueError(
            f"Not all tests for the specified tasks ({task_list}) ran successfully! Error code: {pytest_return_val}"
        )
