"""
RACE: Large-scale ReAding Comprehension Dataset From Examinations
https://arxiv.org/pdf/1704.04683.pdf

RACE is a large-scale reading comprehension dataset with more than 28,000 passages
and nearly 100,000 questions. The dataset is collected from English examinations
in China, which are designed for middle school and high school students. The dataset
can be served as the training and test sets for machine comprehension.

Homepage: https://www.cs.cmu.edu/~glai1/data/race/
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
"""


class RACE(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "race"
    DATASET_NAME = "high"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]
