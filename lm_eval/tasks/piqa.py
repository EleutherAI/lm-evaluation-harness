import numpy as np
from lm_eval.base import MultipleChoiceTask, rf
from ..metrics import mean
from . common import HFTask


class PiQA(HFTask, MultipleChoiceTask):
    DATASET_PATH = "piqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def _convert_standard(self, doc):
        out_doc = {
            "goal": doc["goal"],
            "choices": [doc["sol1"], doc["sol2"]],
            "gold": doc["label"],
        }
        return out_doc

    def _load_docs(self, docs):
        for record in docs:
            yield self._convert_standard(record)

    def training_docs(self):
        docs = super().training_docs()
        return self._load_docs(docs)

    def validation_docs(self):
        docs = super().validation_docs()
        return self._load_docs(docs)

    def test_docs(self):
        docs = super().test_docs()
        return self._load_docs(docs)

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"
