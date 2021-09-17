from . common import HFTask
from lm_eval.base import MultipleChoiceTask


class HeadQA(HFTask, MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "head_qa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        out_doc = {
            "id": doc["qid"],
            "query": "Question: " + doc["qtext"] + "\nAnswer:",
            "choices": [answer["atext"] for answer in doc["answers"]],
            "gold": int(doc["ra"]) - 1,
        }
        return out_doc

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        return doc["query"]
