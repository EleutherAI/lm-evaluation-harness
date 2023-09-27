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
from lm_eval.mixins import MajorityVotingMixin, SymbolicMathMixin

PROMPT=r"""Problem:
Find the value of $x$ that satisfies $\frac{\sqrt{3x+5}}{\sqrt{6x+5}}=\frac{\sqrt{5}}{3}$. Express your answer as a common fraction.

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# Initialize x
x = symbols('x')

# Define the equation
equation = Eq(sqrt(3*x + 5)/sqrt(6*x + 5), sqrt(5)/3)

# Solve for x
answer = solve(equation, x)
```
The imports required for this program are
```
from sympy import symbols, Eq, solve, sqrt
```
I hope my solution is correct.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# Given det(A) = 2 and det(B) = 12
det_A = 2
det_B = 12

# Use the property det(AB) = det(A)*det(B)
det_AB = det_A * det_B

answer = det_AB
```
The imports required for this program are
```

```
I hope my solution is correct. 

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# Calculate the total weight lifted initially, which is 2*20*12 pounds
total_weight = 2 * 20 * 12

# Since Terrell lifts two 15-pound weights, divide the total weight by 2 * 15
repetitions = total_weight / (2*15)

answer = n_value 
```
The imports required for this program are
```

```
I hope my solution is correct. 

Problem:
If Anna flips 8 coins, what is the probability that she gets more heads than tails?

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# There are 2**8 possible outcomes
n = 8
total_outcomes = 2 ** n

# There are binom(n, k) ways to get k heads
favorable_outcomes = 0
for k in range((n // 2) + 1, n + 1):
    favorable_outcomes += math.comb(n, k)
    
probability = favorable_outcomes / total_outcomes

answer = probability
```
The imports required for this program are
```
import math
```
I hope my solution is correct.

Problem:
Evaluate $\left\lceil3\left(6-\frac12\right)\right\rceil$.

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# Calculate 3 * (6 - 1/2)
result = 3 * (6 - 0.5)

# Apply the ceiling function
ceiling_result = math.ceil(result)

answer = ceiling_result
```
The imports required for this program are
```
import math
```
I hope my solution is correct."""

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


class SympyMath(MajorityVotingMixin, SymbolicMathMixin, Task):
    DATASET_PATH = inspect.getfile(lm_eval.datasets.hendrycks_math.hendrycks_math)
    DATASET_NAME = None
    PROMPT = PROMPT
    MAJORITY_VOTING = "majority_voting"
    INVALID_ANSWER = "[invalidanswer]"

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
        example = "Problem:\n" + doc["problem"] + "\n\nYou are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.\n```"

        prompt = self.PROMPT + "\n\n" + example

        return prompt

    @property
    def end_seq(self):
        return "I hope my solution is correct."

    def get_program(self, text: str):
        program = re.search(
                r"(.*?)```",
                text, 
                re.DOTALL,
        )
        if program: 
            body = program.group(1).strip()
        else:
            return "failed to get program"

        imports = re.search(
                r"The imports required for this program are\s```(.*?)```",
                text, 
                re.DOTALL,
        )
        if imports:
            header = imports.group(1).strip()
        else:
            return "failed to get header"
        
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

    def doc_to_target(self, doc):
        raise NotImplementedError
    def doc_to_text(self, doc):
        raise NotImplementedError
 
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
