import numpy as np
from lm_eval.base import MultipleChoiceTask, rf
from ..metrics import mean
from . common import HFTask
from lm_eval.mctask_experimental import MultipleChoiceDoc


class PiQA(HFTask, MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "piqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def _convert_standard(self, doc):
        keys = ['0', '1']
        question = doc["goal"]
        options = [doc["sol1"], doc["sol2"]]
        gold = doc["label"]
        return MultipleChoiceDoc(
            question=question,
            keys=keys,
            options=options,
            gold=gold
        )


