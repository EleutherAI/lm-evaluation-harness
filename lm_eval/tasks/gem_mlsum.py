"""
MLSUM: The Multilingual Summarization Corpus
https://aclanthology.org/2020.emnlp-main.647/

This is the MLSUM subset of the GEM benchmark. MLSUM is the first large-scale MultiLingual SUMmarization dataset.
Obtained from online newspapers, it contains 1.5M+ article/summary pairs in five different languages -- namely, French, German, Spanish, Russian, Turkish.
Together with English newspapers from the popular CNN/Daily mail dataset, the collected data form a large scale multilingual dataset which can enable new research directions for the text summarization community.
We report cross-lingual comparative analyses based on state-of-the-art systems.
These highlight existing biases which motivate the use of a multi-lingual dataset.
Homepage: https://gitlab.lip6.fr/scialom/mlsum_data/-/raw/master/MLSUM/
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@article{scialom2020mlsum,
  title={MLSUM: The Multilingual Summarization Corpus},
  author={Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
  journal={arXiv preprint arXiv:2004.14900},
  year={2020}
}
"""


class GEMMLSUMEsBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "GEM/mlsum"
    DATASET_NAME = "es"

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


class GEMMLSUMEs(GEMMLSUMEsBase):
    """this is for train/validation/test"""

    SPLIT = ""


class GEMMLSUMEsChallgeTestCovid(GEMMLSUMEsBase):
    """this is for challenge_test_covid"""

    SPLIT = "challenge_test_covid"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset[self.SPLIT]


class GEMMLSUMDeBase(PromptSourceTask):

    DATASET_PATH = "GEM/mlsum"
    DATASET_NAME = "de"

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


class GEMMLSUMDe(GEMMLSUMDeBase):
    """this is for train/validation/test"""

    SPLIT = ""


class GEMMLSUMDeChallgeTestCovid(GEMMLSUMDeBase):
    """this is for challenge_test_covid"""

    SPLIT = "challenge_test_covid"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset[self.SPLIT]
