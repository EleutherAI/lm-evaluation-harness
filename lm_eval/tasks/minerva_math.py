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


import sympy
from sympy.parsing.latex import parse_latex

import inspect
import lm_eval.datasets.hendrycks_math.hendrycks_math
from lm_eval.metrics import mean
from lm_eval.base import Task, rf

PROMPT=r"""Problem:
Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain

$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct."""


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
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class MinervaMath(Task):
    DATASET_PATH = inspect.getfile(lm_eval.datasets.hendrycks_math.hendrycks_math)
    DATASET_NAME = None
    MAJORITY_VOTING = "majority_voting"
    SAMPLING_TEMPERATURE = "sampling_temperature"
    EVAL_BATCH_SIZE = "eval_batch_size"
    INVALID_ANSWER="[invalidanswer]"

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

    def doc_to_target(self):
        raise NotImplementedError("MinervaMath has no doc_to_target method.")

    def last_boxed_only_string(self, string):

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

    def remove_boxed(self, s):
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]

        left = "\\boxed{"

        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]

    def _process_doc(self, doc):
        doc["answer"] = self.normalize_final_answer(
                self.remove_boxed(self.last_boxed_only_string(doc["solution"]))
        )
        return doc

    def doc_to_text(self, doc):
        return "Problem:\n" + doc["problem"] + "\n\nSolution:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["problem"]

    def construct_requests(self, doc, ctx, params={}):
        if params == {}:
            return rf.generate(ctx, ["```"])
        
        majority_voting_value = int(params.get(self.MAJORITY_VOTING, 1))
        sampling_temperature_value = float(params.get(self.SAMPLING_TEMPERATURE, 1.0))
        eval_batch_size = params.get(self.EVAL_BATCH_SIZE, None)
        eval_batch_size = int(eval_batch_size) if isinstance(eval_batch_size, str) else eval_batch_size
        generation_params = {
            'num_return_sequences': majority_voting_value,
            'temperature': sampling_temperature_value,
            'num_return_sequences_batch': eval_batch_size
        }
        return rf.generate(ctx, ["```"], generation_params)
    
    def fewshot_context(
            self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        example = self.doc_to_text(doc)
        prompt = PROMPT + "\n\n" + example

        return prompt

    def get_unnormalized_answer(self, text: str):
        match = re.search(
                r'Final Answer: The final answer is(.*?). I hope it is correct.',
                text,
        )
        if match: 
            return match.group(1).strip()
        else:
            return self.INVALID_ANSWER

    def is_equiv(self, x1: str, x2: str):
        """
        x1 and x2 are normalized latex string
        """
        try: 
            with timeout(seconds=5):
                try:
                    parsed_x1 = parse_latex(x1)
                    parsed_x2 = parse_latex(x2)
                except (sympy.parsing.latex.errors.LaTeXParsingError, sympy.SympifyError, TypeError):
                    print(f"couldn't parse one of {x1} or {x2}")
                    return False
            
                try: 
                    diff = parsed_x1 - parsed_x2
                except TypeError:
                    print(f"couldn't subtract {x1} and {x2}")
                    return False

                try:
                    if sympy.simplify(diff)==0:
                        return True
                    else: 
                        return False
                except ValueError:
                    print(f"Had some trouble simplifying when comparing {x1} and {x2}")
        except TimeoutError:
            print(f"Timed out comparing {x1} and {x2}")
            return False
        except Exception as e: 
            print(f"Failed comparing {x1} and {x2} with {e}")
            return False


    def majority_vote(self, candidates):

        # get and normalize all answers
        answers = [
                self.normalize_final_answer(self.get_unnormalized_answer(candidate))
                for candidate in candidates
        ]

        # Count votes for each answer. If two answers are sympy equivalent, 
        # we treat them as the same. 
        answer_votes = {}
        for answer in answers:
            if answer in answer_votes: 
                answer_votes[answer] += 1
            elif answer == self.INVALID_ANSWER:
                pass
            else:
                counted = False
                for ref in answer_votes:
                    if self.is_equiv(answer, ref) and not counted:
                        answer_votes[ref] += 1
                        counted=True

                if not counted: 
                    answer_votes[answer] = 1

        if not answer_votes:
            return self.INVALID_ANSWER, 0, answers

        # Find the argmax and max 
        elected_answer, pass_num = max(answer_votes.items(), key=lambda x: x[1])

        pass_rate = pass_num/len(answers)

        return elected_answer, pass_rate, answers

    def process_results(self, doc, results, params={}):
        candidates = results[0]

        assert isinstance(params, dict)
        
        if params == {}:
            unnormalized_answer = self.get_unnormalized_answer(candidates)
            answer = self.normalize_final_answer(unnormalized_answer)
            answers = [answer]
        elif self.MAJORITY_VOTING in params:
            answer, pass_rate, answers = self.majority_vote(candidates)
        else:
            raise AssertionError

        if self.is_equiv(
            answer, doc["answer"]
        ):
            retval = 1
        else: 
            retval = 0

        if not self.MAJORITY_VOTING:
            pass_rate = retval

        results = {
            "acc": retval,
            "pass_rate": pass_rate,
            "log_pass_rate": math.log(max(pass_rate, 1e-10)),
            "metadata": {
                "selected_answer": answer,
                "candidates": candidates,
                "answers": answers,
            }
        }
        return results

    def aggregation(self):
        return {"acc": mean, "pass_rate": mean, "log_pass_rate": mean}

    def higher_is_better(self):
        return {"acc": True, "pass_rate": True, "log_pass_rate": mean}

    SUBSTITUTIONS = [
        ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), 
        (' ', ''), ('mbox', 'text'), (',\\text{and}', ','), 
        ('\\text{and}', ','), ('\\text{m}', '\\text{}')
    ]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft', 
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 
        'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 
        'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
        '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2', 
        '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}', 
        r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
    ]

    def normalize_final_answer(self, final_answer: str) -> str:
      """
      Normalize a final answer to a quantitative reasoning question.

      Copied character for character from appendix D of Lewkowycz et al. (2022)
      """
      final_answer = final_answer.split('=')[-1]
      
      for before, after in self.SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
      for expr in self.REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')
      
      # Extract answer that is in LaTeX math, is bold, 
      # is surrounded by a box, etc.  
      final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
      final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
      final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
      final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
      final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
      
      # Normalize shorthand TeX:
      #  \fracab -> \frac{a}{b}
      #  \frac{abc}{bef} -> \frac{abc}{bef}
      #  \fracabc -> \frac{a}{b}c
      #  \sqrta -> \sqrt{a}
      #  \sqrtab -> sqrt{a}b
      final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
      final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
      final_answer = final_answer.replace('$', '')
      
      # Normalize 100,000 -> 100000
      if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')
        
      return final_answer


class MinervaMathAlgebraEasy(MinervaMath):
    VERSION = 1
    DATASET_NAME = "algebra"

    def training_docs(self):
        data = map(self._process_doc, self.dataset["train"])
        data = filter(lambda x: x['level'] == 'Level 1', data)
        return data

    def test_docs(self):
        data = map(self._process_doc, self.dataset["train"])
        data = filter(lambda x: x['level'] == 'Level 1', data)
        return data


class MinervaMathAlgebra(MinervaMath):
    VERSION = 1
    DATASET_NAME = "algebra"


class MinervaMathCountingAndProbability(MinervaMath):
    VERSION = 1
    DATASET_NAME = "counting_and_probability"


class MinervaMathGeometry(MinervaMath):
    VERSION = 1
    DATASET_NAME = "geometry"


class MinervaMathIntermediateAlgebra(MinervaMath):
    VERSION = 1
    DATASET_NAME = "intermediate_algebra"


class MinervaMathNumberTheory(MinervaMath):
    VERSION = 1
    DATASET_NAME = "number_theory"


class MinervaMathPrealgebra(MinervaMath):
    VERSION = 1
    DATASET_NAME = "prealgebra"


class MinervaMathPrecalculus(MinervaMath):
    VERSION = 1
    DATASET_NAME = "precalculus"
