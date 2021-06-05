from lm_eval.base import MultipleChoiceTask
from .common import HFTask


class OpenBookQA(HFTask, MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "openbookqa"
    DATASET_NAME = "main"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        out_doc = {
            "id": doc["id"],
            "query": doc["question_stem"],
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D"].index(doc["answerKey"].strip()),
        }
        return out_doc

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def doc_to_text(self, doc):
        return doc["query"]
