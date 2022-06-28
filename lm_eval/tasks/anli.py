"""
Adversarial NLI: A New Benchmark for Natural Language Understanding
https://arxiv.org/pdf/1910.14599.pdf

Adversarial NLI (ANLI) is a dataset collected via an iterative, adversarial
human-and-model-in-the-loop procedure. It consists of three rounds that progressively
increase in difficulty and complexity, and each question-answer includes annotator-
provided explanations.

Homepage: "https://github.com/facebookresearch/anli"
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{nie-etal-2020-adversarial,
    title = "Adversarial {NLI}: A New Benchmark for Natural Language Understanding",
    author = "Nie, Yixin  and
      Williams, Adina  and
      Dinan, Emily  and
      Bansal, Mohit  and
      Weston, Jason  and
      Kiela, Douwe",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
"""


class ANLIBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "anli"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train_r" + str(self.SPLIT)]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["dev_r" + str(self.SPLIT)]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test_r" + str(self.SPLIT)]


class ANLIRound1(ANLIBase):
    SPLIT = 1


class ANLIRound2(ANLIBase):
    SPLIT = 2


class ANLIRound3(ANLIBase):
    SPLIT = 3
