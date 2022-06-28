"""
A dataset of approximately 200K news headlines from the year 2012 to 2018 collected from HuffPost.

Homepage: https://www.kaggle.com/datasets/rmisra/news-category-dataset
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """\
@book{book,
  author = {Misra, Rishabh and Grover, Jigyasa},
  year = {2021},
  month = {01},
  pages = {},
  title = {Sculpting Data for ML: The first act of Machine Learning},
  isbn = {978-0-578-83125-1}
}
@dataset{dataset,
  author = {Misra, Rishabh},
  year = {2018},
  month = {06},
  pages = {},
  title = {News Category Dataset},
  doi = {10.13140/RG.2.2.20331.18729}
}
"""


class HuffPost(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "khalidalt/HuffPost"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
