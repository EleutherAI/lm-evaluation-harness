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
from sympy.core.sympify import SympifyError
from sympy.parsing.latex import parse_latex

from lm_eval.utils import timeout
from lm_eval.base import rf

class MajorityVotingMixin:
    """
    Majority voting for an arbitrary definition of equivalence.

    Also enables support for temperature and top-p sampling. 

    The `majority_vote` function should likely be called by the subclass in `Task.process_results()`.
    The `construct_requests` method works with no code changes to the subclass, 
    but requires passing the `--description_dict_path` cli argument
    """
    MAJORITY_VOTING = "majority_voting"
    SAMPLING_TEMPERATURE = "sampling_temperature"
    TOP_P = "top_p"
    EVAL_BATCH_SIZE = "eval_batch_size"
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
            pass_rate: float, proportion of `sampled_answers` equivalent to `correct_answer`
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

    def construct_requests(self, doc, ctx, params={}):
        if params == {}:
            if isinstance(self.end_seq, str):
                return rf.generate(ctx, [self.end_seq])
            else:
                return rf.generate(ctx, self.end_seq)
        
        majority_voting_value = int(params.get(self.MAJORITY_VOTING, 1))
        sampling_temperature_value = float(params.get(self.SAMPLING_TEMPERATURE, 1.0))
        top_p = float(params.get(self.TOP_P, 1.0))
        eval_batch_size = params.get(self.EVAL_BATCH_SIZE, None)
        eval_batch_size = int(eval_batch_size) if isinstance(eval_batch_size, str) else eval_batch_size
        generation_params = {
            'num_return_sequences': majority_voting_value,
            'temperature': sampling_temperature_value,
            'top_p': top_p,
            'num_return_sequences_batch': eval_batch_size
        }
        if isinstance(self.end_seq, str):
            return rf.generate(ctx, [self.end_seq], generation_params)
        else:
            return rf.generate(ctx, self.end_seq, generation_params)

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
            SympifyError,
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
                except (SympifyError, ValueError, TypeError) as e:
                    print(
                        f"Couldn't subtract {x1} and {x2} with exception {e}"
                    )
                    return False

                try:
                    if sympy.simplify(diff) == 0:
                        return True
                    else:
                        return False
                except (SympifyError, ValueError, TypeError) as e:
                    print(f"Failed to simplify {x1}-{x2} with {e}")
                    return False
        except TimeoutError as e:
            print(f"Timed out comparing {x1} and {x2}")
            return False
        except Exception as e:
            print(f"failed on unrecognized exception {e}")
            return False

    def is_tex_equiv(self, x1: str, x2: str, time_limit=5) -> bool:
        """
        Determines whether two (ideally normalized using `normalize_text`) TeX expressions are equal.
        """
        return self.is_exp_equiv(self.parse_tex(x1), self.parse_tex(x2), time_limit=time_limit)
