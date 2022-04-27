from lm_eval.base import PromptSourceTask


class WebNLG(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "GEM/web_nlg"
    DATASET_NAME = "en"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def stopping_criteria(self):
        return '*'

    def max_generation_length(self):
        return 250

