"""
"Training Verifiers to Solve Math Word Problems"
https://arxiv.org/abs/2110.14168

State-of-the-art language models can match human performance on many tasks, but
they still struggle to robustly perform multi-step mathematical reasoning. To
diagnose the failures of current models and support research, we introduce GSM8K,
a dataset of 8.5K high quality linguistically diverse grade school math word problems.
We find that even the largest transformer models fail to achieve high test performance,
despite the conceptual simplicity of this problem distribution.

NOTE: See the official implementation of the task:
    https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py
for how to make use of the dataset's calculator annotations in your language
model's sample/generation function.

Homepage: https://github.com/openai/grade-school-math
"""
import re
from lm_eval.base import Task, rf
from lm_eval.mixins import MajorityVotingMixin
from lm_eval.metrics import mean, weighted_perplexity


_CITATION = """
@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


class GradeSchoolMath8K(MajorityVotingMixin, Task):
    VERSION = 0
    DATASET_PATH = "gsm8k"
    DATASET_NAME = "main"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        raise NotImplementedError

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    @property
    def end_seq(self):
        return "\n\n"

    def _extract_answer(self, completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    def process_results(self, doc, results, params):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        completion = results[0]

        gold = self._extract_answer(doc["answer"])

        if self.MAJORITY_VOTING not in params:
            answer = self._extract_answer(completion)

            if answer==gold:
                acc = 1
            else:
                acc = 0

            pass_rate = acc
        else:
            answers = [self._extract_answer(x) for x in completion]
            answers = [x for x in answers if x!=INVALID_ANS]

            acc, pass_rate, votes = self.majority_vote(
                    answers,
                    correct_answer=gold
            )

            if votes:
                answer = votes[0][0]
            else:
                answer = INVALID_ANS


        return_dict = {
                "acc": acc,
                "pass_rate": pass_rate,
                "metadata": {"completion": completion, "selected_answer": answer}
        }

        if self.MAJORITY_VOTING in params:
            return_dict['metadata']['votes'] = votes

        return return_dict
    
    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean, "pass_rate": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True, "pass_rate": True}
