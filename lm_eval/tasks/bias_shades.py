"""
Multilingual dataset for measuring social biases in language models.
French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English
"""
from lm_eval.base import PromptSourceTask


_CITATION = """"""


class ShadesEnglish(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "english"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class ShadesFrench(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "french"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class ShadesHindi(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "hindi"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class ShadesArabic(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "arabic"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class ShadesSpanish(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "spanish"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
