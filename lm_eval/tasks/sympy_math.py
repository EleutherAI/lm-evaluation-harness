"""
Measuring Mathematical Problem Solving With the MATH Dataset
https://arxiv.org/pdf/2103.03874.pdf

Math is a dataset of 12,500 challenging competition mathematics problems. Each
problem in Math has a full step-by-step solution which can be used to teach
models to generate answer derivations and explanations.

Homepage: https://github.com/hendrycks/math
"""
import re
import math
import code
import signal
from abc import ABC

import sympy

import inspect
import lm_eval.datasets.hendrycks_math.hendrycks_math
from lm_eval.metrics import mean
from lm_eval.base import Task, rf
from lm_eval.utils import MajorityVotingMixin, SymbolicMathMixin
from lm_eval.tasks.math_tasks import SymbolicMathTask


CODE_PROMPT=r"""Problem:
Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
```
# Initialize variable x as a symbol
x = Symbol('x')

# The square root of (x-2) requires x-2 >= 0
domain1 = solveset(x - 2 >= 0, x, domain=S.Reals)

# The square root of (5-x) requires 5-x > 0 (can't be zero because it's in the denominator)
domain2 = solveset(5 - x > 0, x, domain=S.Reals)

# Step 4: Intersect both domains to find the final domain of the expression
final_domain = domain1.intersect(domain2)

answer = final_domain
```
The imports required for this program are
```
from sympy import Symbol, solveset, S
```
I hope my solution is correct.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution:
```
# Given det(A) = 2 and det(B) = 12
det_A = Integer(2)
det_B = Integer(12)

# Use the property det(AB) = det(A)*det(B)
det_AB = det_A * det_B

answer = det_AB
```
The imports required for this program are
```
from sympy import Integer
```
I hope my solution is correct. 

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
```
# Initialize n as a symbol, representing the number of times Terrell must lift the 15-pound weights
n = Symbol('n')

# Calculate the total weight lifted initially, which is 2*20*12 pounds
total_weight = 2 * 20 * 12

# Set up the equation for total weight lifted with 15-pound weights
equation = 2 * 15 * n - total_weight

# Step 4: Solve for n
n_value = solve(equation, n)[0]

answer = n_value 
```
The imports required for this program are
```
from sympy import Symbol, solve
```
I hope my solution is correct. 

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
```
# Initialize symbols for a, b, x, y
a, b = symbols('a b')
x, y = symbols('x y')

# Since b!=0, we can obtain a/b by dividing the first equation by the second
a_over_b = (6*x - 4*y) / (6*y - 9*x)

# Simplify the expression
a_over_b = simplify(a_over_b)

answer = a_over_b
```
The imports required for this program are
```
from sympy import symbols, simplify
```
I hope my solution is correct. 
"""

_CITATION = """
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the Math Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
@misc{lewkowycz2022solving,
      title={Solving Quantitative Reasoning Problems with Language Models}, 
      author={Aitor Lewkowycz and Anders Andreassen and David Dohan and Ethan Dyer and Henryk Michalewski and Vinay Ramasesh and Ambrose Slone and Cem Anil and Imanol Schlag and Theo Gutman-Solo and Yuhuai Wu and Behnam Neyshabur and Guy Gur-Ari and Vedant Misra},
      year={2022},
      eprint={2206.14858},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class SympyMath(Task, MajorityVotingMixin, SymoblicMathMixin):
    DATASET_PATH = inspect.getfile(lm_eval.datasets.hendrycks_math.hendrycks_math)
    DATASET_NAME = None
    PROMPT = CODE_PROMPT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("WARNING: Ignores --num-fewshot argument and uses a fixed prompt")

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def fewshot_context(
            self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        example = "Problem:\n" + doc["problem"] + "\n\nSolution:```"
        prompt = self.PROMPT + "\n\n" + example

        return prompt

    @property
    def end_seq(self):
        return "I hope my solution is correct."

    def _get_program(self, text: str):
        program = re.search(
                r"```(.*?)```"
                text, 
                re.MULTILINE,
        )
        if program: 
            body = match.group(1).strip()
        else:
            return self.INVALID_ANSWER

        imports = re.search(
                r"The imports required for this program are\s```(.*?)```",
                text, 
                re.MULTILINE,
        )
        if imports:
            header = match.group(1).strip()
        else:
            return self.INVALID_ANSWER
        
        code = header + "\n\n" + body
        return code


    def _process_doc(self, doc):
        doc["answer"] = self.normalize_tex(
                self._remove_boxed(self._last_boxed_only_string(doc["solution"]))
        )
        return doc

    def _last_boxed_only_string(self, string):
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

    def _remove_boxed(self, s):
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]

        left = "\\boxed{"

        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]

    def construct_requests(self, doc, ctx, params={}):
        if params == {}:
            return rf.generate(ctx, [self.end_seq])
        
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
        return rf.generate(ctx, [self.end_seq], generation_params)

    def process_results(self, doc, results, params={}):
        candidates = results[0]

        if self.MAJORITY_VOTING in params:
            programs = [self.get_program(sample) for sample in candidates]
        else: 
            programs = self.get_program(candidates)

        assert isinstance(params, dict)
        
        results = {
            "metadata": {
                "unprocessed_samples": candidates,
                "extracted_programs": programs,
            }
        }
        return results

    def aggregation(self):
        return {}

    def higher_is_better(self):
        return {}
 
class SympyMathAlgebraEasy(SympyMath):
    VERSION = 1
    DATASET_NAME = "algebra"

    def training_docs(self):
        data = map(self._process_doc, self.dataset["train"])
        data = filter(lambda x: x['level'] == 'Level 1', data)
        return data

    def test_docs(self):
        data = map(self._process_doc, self.dataset["test"])
        data = filter(lambda x: x['level'] == 'Level 1', data)
        return data


class SympyMathAlgebra(SympyMath):
    VERSION = 1
    DATASET_NAME = "algebra"


class SympyMathCountingAndProbability(SympyMath):
    VERSION = 1
    DATASET_NAME = "counting_and_probability"


class SympyMathGeometry(SympyMath):
    VERSION = 1
    DATASET_NAME = "geometry"


class SympyMathIntermediateAlgebra(SympyMath):
    VERSION = 1
    DATASET_NAME = "intermediate_algebra"


class SympyMathNumberTheory(SympyMath):
    VERSION = 1
    DATASET_NAME = "number_theory"


class SympyMathPrealgebra(SympyMath):
    VERSION = 1
    DATASET_NAME = "prealgebra"


class SympyMathPrecalculus(SympyMath):
    VERSION = 1
    DATASET_NAME = "precalculus"
