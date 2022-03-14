from ast import Mult
import re
from lm_eval.base import MultipleChoiceTask
from . common import HFTask
from lm_eval.mctask_experimental import MultipleChoiceDoc


class HellaSwag(HFTask, MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "hellaswag"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub('\\[.*?\\]', '', text)
        text = text.replace("  ", " ")
        return text

    def _convert_standard(self, doc):
        question = self.preprocess(doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
        options = [self.preprocess(ending) for ending in doc['endings']]
        gold = int(doc["label"])
        keys = ["A", "B", "C", "D"]
        context = self.preprocess(doc['activity_label'])
        return MultipleChoiceDoc(
            question=question,
            options=options,
            gold=gold,
            keys=keys,
            context=context
        )
