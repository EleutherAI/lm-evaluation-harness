from lm_eval.base import MultipleChoiceTask
from .common import HFTask


class OpenBookQA(HFTask, MultipleChoiceTask):
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
        # TODO: figure out fewshot description
        return ""

    def doc_to_text(self, doc):
        return doc["query"]
