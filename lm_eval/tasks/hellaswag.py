import re
from lm_eval.base import MultipleChoiceTask
from . common import HFTask


class HellaSwag(HFTask, MultipleChoiceTask):
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
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": self.preprocess(doc['activity_label'] + ': ' + ctx),
            "choices": [self.preprocess(ending) for ending in doc['endings']],
            "gold": int(doc['label']),
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

    def fewshot_description(self):
        return "Label for the relevant action: Sentences describing the " \
            "context, with an incomplete sentence trailing\nanswer that " \
            "plausibly completes the situation."

    def doc_to_text(self, doc):
        return doc["query"]
