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
import code
from lm_eval.base import Task, rf
from lm_eval.mixins import MajorityVotingMixin
from lm_eval.metrics import mean, weighted_perplexity

PROMPT=r"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.  The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."""


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
SAMPLE_RE = re.compile(r"The answer is (\d+)")
INVALID_ANS = "[invalid]"


class GradeSchoolMath8K(MajorityVotingMixin, Task):
    VERSION = 0
    DATASET_PATH = "gsm8k"
    DATASET_NAME = "main"
    PROMPT = PROMPT

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
        return "Q: " + doc["question"] + "\nA:"

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    @property
    def end_seq(self):
        return "\n\n"

    def fewshot_context(
            self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        example = self.doc_to_text(doc)
        prompt = self.PROMPT + "\n\n" + example

        return prompt

    def _extract_answer(self, completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    def _extract_answer_from_sample(self, completion):
        match = SAMPLE_RE.search(completion)
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
            answer = self._extract_answer_from_sample(completion)

            if answer==gold:
                acc = 1
            else:
                acc = 0

            pass_rate = acc
        else:
            answers = [self._extract_answer_from_sample(x) for x in completion]
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
