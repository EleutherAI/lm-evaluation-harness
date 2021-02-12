from . common import HFTask
from lm_eval.base import mean, rf, MultipleChoiceTask


class MathQA(HFTask, MultipleChoiceTask):
    DATASET_PATH = "math_qa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):

        out_doc = {
            "query": "Question: " + doc['Problem'] +" "+ doc["options"] + "\nAnswer:",
            "choices": ['a', 'b', 'c', 'd', 'e'],
            "gold": ['a', 'b', 'c', 'd', 'e'].index(doc['correct']),
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

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        return doc["query"]
