import re
from lm_eval.base import MultipleChoiceTask
from . common import HFTask


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

        answer_idx = ['a', 'b', 'c', 'd', 'e'].index(doc['correct'])
        choices = [c[4:].rstrip(" ,") for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc['options'])]

        out_doc = {
            "query": "Question: " + doc['Problem'] +"\nAnswer:",
            "choices": choices,
            "gold": answer_idx,
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
