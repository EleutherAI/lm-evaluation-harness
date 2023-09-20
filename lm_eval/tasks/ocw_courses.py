"""
OCWCourses
https://arxiv.org/pdf/2103.03874.pdf
"""
import re
import math
import code
import signal
from abc import ABC

import inspect
from lm_eval.metrics import mean
from lm_eval.base import Task, rf
from lm_eval.tasks.math_tasks import SymbolicMathTask


NL_PROMPT=r"""Problem:                                                                                
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


class OCWCourses(SymbolicMathTask):
    DATASET_PATH = "open-web-math/ocwcourses"
    DATASET_NAME = None
    PROMPT = NL_PROMPT
    VERSION = 1

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
                r'Final answer: The final answer is(.*?). I hope it is correct.',
                text,
        )
        if match: 
            ans = match.group(1).strip()
        else:
            ans = self.INVALID_ANSWER 

        print(f"\n EXTRACTED_ANSWER: {ans}")

        return ans

    def _doc_to_text(self, doc):
        return "Problem:\n" + doc["problem"] + "\nSolution:"
