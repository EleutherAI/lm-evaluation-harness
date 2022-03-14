from lm_eval.base import MultipleChoiceTask
from . common import HFTask
from lm_eval.mctask_experimental import MultipleChoiceDoc


class ARCEasy(HFTask, MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Easy"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        question = doc["question"]
        keys = ["A", "B", "C", "D", "E"]
        options = doc["choices"]["text"]
        while len(options) < len(keys):
            options.append("")
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        gold = ["A", "B", "C", "D", "E"].index(doc["answerKey"])
        return MultipleChoiceDoc(
            question=question,
            options=options,
            gold=gold,
            keys=keys,
        )

class ARCChallenge(ARCEasy):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"
