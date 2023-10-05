"""
SAT Math May 2023 questions that do not have figures.

We use the version of the dataset found in the Huggingface dataset `mcaleste/sat_multiple_choice_math_may_23`.

Our prompt is taken from from appendix G of Lewkowycz et al. (2022). 
"""

from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from lm_eval.mixins import MajorityVotingMixin

import re

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



MCQA_PROMPT = r"""Problem:
Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.
What of the following is the right choice? Explain your answer.
(A) [-5,-2), (B) [2,5), (C) [-2,-5), (D) [5,2)
Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \
ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.
Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is (B). I hope it is correct.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$
What of the following is the right choice? Explain your answer.
(A) 14, (B) 4, (C) 2, (D) 24
Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
Final Answer: The final answer is (D). I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times \
must Terrell lift them in order to lift the same total weight?
What of the following is the right choice? Explain your answer.
(A) 12, (B) 20, (C) 16, (D) 15
Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight. \
If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ \
pounds of weight. Equating this to 480 pounds, we can solve for $n$: \begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
Final Answer: The final answer is (C). I hope it is correct.

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b},$ assuming $b$ is
nonzero.
What of the following is the right choice? Explain your answer.
(A) $-\frac{2}{3}$, (B) $\frac{2}{3}$, (C) $\frac{1}{3}$, (D) $\frac{4}{9}$
Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain
$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is (A). I hope it is correct."""


class MathSATCoT(MajorityVotingMixin, Task):
    VERSION = 0
    DATASET_PATH = "mcaleste/sat_multiple_choice_math_may_23"
    DATASET_NAME = None

    ANS_RE = re.compile(r"Final Answer: The final answer is \([ABCD]\). I hope it is correct.")
    INVALID_ANS = "[not found]"

    def __init__(self):
        print("WARNING: math_sat_cot ignores --num-fewshot argument and uses a fixed prompt")
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["train"])
    
    def fewshot_context(
            self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        return doc["query"]

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Problem: <prompt>
            What of the following is the right choice? Explain you answer.
            (A) <choice1>, (B) <choice2>, (C) <choice3>, (D) <choice4>
            Solution:
            """
            prompt = MCQA_PROMPT + "\n\n" + "Problem:\n" + doc["Question"] + "\nWhat of the following is the right choice? Explain your answer.\n"
            prompt += doc["Possible Answers"]
 #                 [f"{key} {choice}" for key, choice in zip(keys, doc["Possible Answers"])]
 #            )
            prompt += "\nSolution:"
            return prompt
        
        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": doc["Possible Answers"],
            "gold": "(" + doc["Answer"] + ")"
        }

    def doc_to_text(self, doc):
        return doc["query"]

    @property
    def end_seq(self):
        return ["\n\n", "Problem:"]

    def process_results(self, doc, results, params={}):
        candidates = results[0]
        assert isinstance(params, dict)
        if params == {}:
            completion = self._extract_answer(candidates)
            acc = self._is_correct(completion, doc['gold'])
            pass_rate = acc
        elif self.MAJORITY_VOTING in params:
            acc, pass_rate, votes = self.majority_vote(
                    [self._extract_answer(c) for c in candidates if self._extract_answer(c)!=self.INVALID_ANS],
                    correct_answer=doc['gold'],
                    # is_equiv=self._is_correct, this line commented out since is_equiv assumed to be symmetric
            )
            if votes:
                completion = votes[0][0]
            else:
                completion = self.INVALID_ANS
        else:
            raise AssertionError

        return_dict = {
            "acc": acc,
            "pass_rate": pass_rate,
            "metadata": {
                "selected_answer": completion,
                "candidates": candidates
            }
        }

        if self.MAJORITY_VOTING in params:
            return_dict['metadata']['votes'] = votes

        return return_dict
        
    def _extract_answer(self, completion):
        match = self.ANS_RE.search(completion)
        if match is not None:
            match_str = match.group(0)
            match_str = match_str.lstrip("Final Answer: The final answer is ").rstrip(". I hope it is correct.")
            print(match_str)
            return match_str
        else:
            return self.INVALID_ANS  

    def _is_correct(self, completion, answer):
        gold = answer
        assert gold != self.INVALID_ANS, "No ground truth answer found in the document."
        return completion == gold

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't

        raise NotImplementedError
    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        raise NotImplementedError("Should not rely on doc_to_target for pure-zeroshot Minerva-MMLU(STEM)")

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def aggregation(self):
        return {"acc": mean, "pass_rate": mean}

    def higher_is_better(self):
        return {"acc": True, "pass_rate": True}
    
