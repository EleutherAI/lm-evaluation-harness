"""
OCWCourses
https://arxiv.org/pdf/2103.03874.pdf
"""
import re
import math
import code
import signal
from abc import ABC
from typing import Optional

import inspect
from lm_eval.metrics import mean
from lm_eval.base import Task, rf
from lm_eval.mixins import MajorityVotingMixin, SymbolicMathMixin

from sympy.parsing.latex import parse_latex

import numpy as np

NL_PROMPT = r"""Problem:                                                                                
Subproblem 0: What is the net charge of arginine in a solution of $\mathrm{pH} 1.0$? 
Please format your answer as +n or -n.                           
Solution:
The answer is +2.                                                     
Final answer: The final answer is +2. I hope it is correct.

Problem:
Subproblem 0: Let $z = 1 + \sqrt{3} i$. Find $a, b$ that satisfy the equation 
$z^4 = a + bi$. Express your answer as the ordered pair $(a,b)$.         
Solution:
$z^{4}$ has argument $4 \pi / 3$ and radius 16 , so it's equal to $-8-8 \sqrt{3} i$. 
Thus $a = -8, b = -8\sqrt 3$, and our answer is $\boxed{(-8, -8\sqrt{3})}$.
Final answer: The final answer is (-8, -8\sqrt{3}). I hope it is correct.

Problem:
Preamble: For each Laplace Transform \(Y(s)\), find the function \(y(t)\):
Subproblem 0: 
\[Y(s)=\boxed{\frac{1}{(s+a)(s+b)}}\]
Solution:
We can simplify with partial fractions:
\[Y(s)=\\frac{1}{(s+a)(s+b)}=\\frac{C}{s+a}+\\frac{D}{s+b}\]\nfind the constants 
\(C\) and \(D\) by setting \(s=-a\) and \(s=-b\)
\[
  \begin{aligned}
  \frac{1}{(s+a)(s+b)} &=\\frac{C}{s+a}+\\frac{D}{s+b} \\\\
  1 &=C(s+b)+D(s+a) \\
  C &=\\frac{1}{b-a} \\
  D &=\\frac{1}{a-b}
  \end{aligned}
\]
therefore
\[\nY(s)=\frac{1}{b-a} \frac{1}{s+a}-\frac{1}{b-a} \frac{1}{s+b}
\]
By looking up the inverse Laplace Transform of \(\frac{1}{s+b}\), we find the total 
solution \(y(t)\)
\[
  y(t)=\boxed{\frac{1}{b-a}\left(e^{-a t}-e^{-b t}\right)}
\].
Final answer: The final answer is \[\frac{1}{b-a}\left(e^{-a t}-e^{-b t}\right)\]. I hope it is correct.

Problem:
Preamble: The following subproblems refer to the differential equation 
$\ddot{x}+b \dot{x}+x=0$.
Subproblem 0: What is the characteristic polynomial $p(s)$ of 
$\ddot{x}+b \dot{x}+x=0$?
Solution:
The characteristic polynomial is $p(s)=\\boxed{s^{2}+b s+1}$.
Final answer: The final answer is $s^{2}+b s+1$. I hope it is correct."""


_CITATION = """
@misc{lewkowycz2022solving,
      title={Solving Quantitative Reasoning Problems with Language Models}, 
      author={Aitor Lewkowycz and Anders Andreassen and David Dohan and Ethan Dyer and Henryk Michalewski and Vinay Ramasesh and Ambrose Slone and Cem Anil and Imanol Schlag and Theo Gutman-Solo and Yuhuai Wu and Behnam Neyshabur and Guy Gur-Ari and Vedant Misra},
      year={2022},
      eprint={2206.14858},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class OCWCourses(MajorityVotingMixin, SymbolicMathMixin, Task):
    DATASET_PATH = "open-web-math/ocwcourses"
    DATASET_NAME = None
    PROMPT = NL_PROMPT
    VERSION = 1
    INVALID_ANSWER = "[invalidanswer]"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("WARNING: Ignores --num-fewshot argument and uses a fixed prompt")

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return NotImplemented

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return self.dataset["test"]

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        example = self._doc_to_text(doc)
        prompt = self.PROMPT + "\n\n" + example

        return prompt

    @property
    def end_seq(self):
        return "I hope it is correct."

    def get_unnormalized_answer(self, text: str):
        text += self.end_seq

        print(f"\n\n### TEXT TO RE:\n{text}")
        match = re.search(
            r"Final answer: The final answer is(.*?). I hope it is correct.", text,
        )
        if match:
            ans = match.group(1).strip()
        else:
            ans = self.INVALID_ANSWER

        print(f"\n EXTRACTED_ANSWER: {ans}")

        return ans

    def _doc_to_text(self, doc):
        return "Problem:\n" + doc["problem"] + "\nSolution:"

    def doc_to_target(self):
        raise NotImplementedError("SymbolicMathTask has no doc_to_target method.")

    def doc_to_text(self, doc):
        raise NotImplementedError("SymbolicMathTask does not implement doc_to_text")

    def should_decontaminate(self):
        return False

    def process_results(self, doc, results, params={}):
        candidates = results[0]

        assert isinstance(params, dict)

        ref = doc['answer']

        try:
            float(ref)
            normalize_fn = self.normalize_numeric
            is_equiv = self.numeric_equality
            answer_type = "numeric"
        except ValueError:
            if "=" in ref:
                normalize_fn = self.normalize_symbolic_equation
                is_equiv = lambda x, y: x==y
                answer_type = "equation"
            else:
                normalize_fn = self.normalize_tex
                is_equiv = self.is_tex_equiv
                answer_type = "expression"

        correct_answer = normalize_fn(ref)

        if self.MAJORITY_VOTING not in params:
            unnormalized_answer = self.get_unnormalized_answer(candidates)

            model_answer = normalize_fn(unnormalized_answer)

            if unnormalized_answer == self.INVALID_ANSWER:
                acc = 0
            elif model_answer == self.INVALID_ANSWER:
                acc = 0
            elif is_equiv(model_answer, correct_answer):
                acc = 1
            else:
                acc = 0

            pass_rate = acc
        else:
            answers = [
                normalize_fn(self.get_unnormalized_answer(candidate))
                for candidate in candidates
                if self.get_unnormalized_answer(candidate) != self.INVALID_ANSWER
                and normalize_fn(self.get_unnormalized_answer(candidate)) != self.INVALID_ANSWER
            ]

            acc, pass_rate, votes = self.majority_vote(
                answers, correct_answer=correct_answer, is_equiv=is_equiv,
            )
            if votes:
                model_answer = votes[0][0]
            else:
                model_answer = self.INVALID_ANSWER

        results = {
            "acc": acc,
            "pass_rate": pass_rate,
            "metadata": {
                "selected_answer": model_answer, 
                "unprocessed_answers": candidates,
                "answer_type": answer_type,
            },
        }

        if self.MAJORITY_VOTING in params:
            results["metadata"]["votes"] = votes

        return results

    def aggregation(self):
        return {"acc": mean, "pass_rate": mean}

    def higher_is_better(self):
        return {"acc": True, "pass_rate": True}

    def normalize_numeric(self, s):
        if s is None:
            return None
        for unit in [
            "eV",
            " \\mathrm{~kg} \\cdot \\mathrm{m} / \\mathrm{s}",
            " kg m/s",
            "kg*m/s",
            "kg",
            "m/s",
            "m / s",
            "m s^{-1}",
            "\\text{ m/s}",
            " \\mathrm{m/s}",
            " \\text{ m/s}",
            "g/mole",
            "g/mol",
            "\\mathrm{~g}",
            "\\mathrm{~g} / \\mathrm{mol}",
            "W",
            "erg/s",
            "years",
            "year",
            "cm",
        ]:
            s = s.replace(unit, "")
            s = s.strip()
        for maybe_unit in ["m", "s", "cm"]:
            s = s.replace("\\mathrm{" + maybe_unit + "}", "")
            s = s.replace("\\mathrm{~" + maybe_unit + "}", "")
            s = s.strip()
        s = s.strip("$")
        try:
            return float(eval(s))
        except:
            try:
                expr = parse_latex(s)
                if expr.is_number:
                    return float(expr)
                return self.INVALID_ANSWER
            except:
                return self.INVALID_ANSWER


    def numeric_equality(self, n1, n2, threshold=0.01):
        if n1 is None or n2 is None:
            return False
        if np.isclose(n1, 0) or np.isclose(n2, 0) or np.isclose(n1 - n2, 0):
            return np.abs(n1 - n2) < threshold * (n1 + n2) / 2
        else:
            return np.isclose(n1, n2)


    def normalize_symbolic_equation(self, s: Optional[str]):
        if not isinstance(s, str):
            return self.INVALID_ANSWER
        if s.startswith("\\["):
            s = s[2:]
        if s.endswith("\\]"):
            s = s[:-2]
        s = s.replace("\\left(", "(")
        s = s.replace("\\right)", ")")
        s = s.replace("\\\\", "\\")
        if s.startswith("$") or s.endswith("$"):
            s = s.strip("$")
        try:
            maybe_expression = parse_latex(s)
            if not isinstance(maybe_expression, sympy.core.relational.Equality):
                # we have equation, not expression
                return self.INVALID_ANSWER
            else:
                return maybe_expression
        except:
            return self.INVALID_ANSWER


    def normalize_symbolic_expression(self, s: Optional[str]):
        if not isinstance(s, str):
            return self.INVALID_ANSWER
        if s.startswith("\\["):
            s = s[2:]
        if s.endswith("\\]"):
            s = s[:-2]
        s = s.replace("\\left(", "(")
        s = s.replace("\\right)", ")")
        s = s.replace("\\\\", "\\")
        if s.startswith("$") or s.endswith("$"):
            s = s.strip("$")
        try:
            maybe_expression = parse_latex(s)
            if isinstance(maybe_expression, sympy.core.relational.Equality):
                # we have equation, not expression
                return self.INVALID_ANSWER
            if isinstance(maybe_expression, sympy.logic.boolalg.BooleanFalse):
                return self.INVALID_ANSWER
            else:
                return maybe_expression
        except:
            return self.INVALID_ANSWER
