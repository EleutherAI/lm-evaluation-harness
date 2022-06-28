"""
Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization
https://arxiv.org/pdf/1808.08745.pdf

The dataset is for the task of abstractive summarization in its extreme form, its about summarizing a document in a single sentence. It introduces extreme summarization, a new single-document summarization task which does not favor extractive strategies and calls for an abstractive modeling approach. The idea is to create a short, one-sentence news summary answering the question "What is the article about?".

This particularly uses the dataset that is part of the GEM benchmark
Homepage: https://github.com/EdinburghNLP/XSum
The GEM Benchmark: Natural Language Generation, its Evaluation and Metrics
https://arxiv.org/pdf/2102.01672v3.pdf
Write a Short Description of the task.
Homepage: https://gem-benchmark.com/data_cards/XSum
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@InProceedings{xsum-emnlp,
  author =      "Shashi Narayan and Shay B. Cohen and Mirella Lapata",
  title =       "Don't Give Me the Details, Just the Summary! {T}opic-Aware Convolutional Neural Networks for Extreme Summarization",
  booktitle =   "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing ",
  year =        "2018",
  address =     "Brussels, Belgium",
}
"""


class GEMXSUMBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "GEM/xsum"
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
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self):
        return 64


class GEMXSUM(GEMXSUMBase):
    """this is for train/validation/test"""

    SPLIT = ""


class GEMXSUMChallgeSample(GEMXSUMBase):
    """this is for challenge_train_sample/challenge_validation_sample"""

    SPLIT = "challenge_sample"

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["challenge_train_sample"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["challenge_validation_sample"]


class GEMXSUMChallgeTestBacktranslation(GEMXSUMBase):
    """this is for challenge_test_backtranslation"""

    SPLIT = "challenge_test_backtranslation"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset[self.SPLIT]


class GEMXSUMChallgeTestBFP02(GEMXSUMBase):
    """this is for challenge_test_bfp_02"""

    SPLIT = "challenge_test_bfp_02"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset[self.SPLIT]


class GEMXSUMChallgeTestBFP05(GEMXSUMBase):
    """this is for challenge_test_bfp_05"""

    SPLIT = "challenge_test_bfp_05"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset[self.SPLIT]


class GEMXSUMChallgeTestNopunc(GEMXSUMBase):
    """this is for challenge_test_nopunc"""

    SPLIT = "challenge_test_nopunc"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset[self.SPLIT]


class GEMXSUMChallgeTestCovid(GEMXSUMBase):
    """this is for challenge_test_covid"""

    SPLIT = "challenge_test_covid"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset[self.SPLIT]
