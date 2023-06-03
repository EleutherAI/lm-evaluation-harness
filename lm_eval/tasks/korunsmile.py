"""
 Korean UnSmile Dataset
 
 Github: https://github.com/smilegate-ai/korean_unsmile_dataset
"""

import numpy as np
from lm_eval.base import MultipleChoiceTask
from lm_eval.metrics import macro_f1_score

_CITATION = """
@misc{SmilegateAI2022KoreanUnSmileDataset,
  title         = {Korean UnSmile dataset: Human-annotated Multi-label Korean Hate Speech Dataset},
  author        = {Seonghyun Kim},
  year          = {2022},
  howpublished  = {https://github.com/smilegate-ai/korean_unsmile_dataset},
}
"""

def multilable_to_multiclass(label: list):
    """
    0 = 혐오
    1 = 악플
    2 = 양호
    """
    assert type(label[0]) == int
    _id = np.argmax(label)

    if _id == 8:
        return 1
    elif _id == 9:
        return 2
    else:
        return 0


class KorUnSmile(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "smilegate-ai/kor_unsmile"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc,self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc,self.dataset["valid"])

    def _process_doc(self, doc):
        out_doc = {
            "title": doc["문장"],
            "choices": ["혐오", "악플", "양호"],
            "gold": multilable_to_multiclass(doc["labels"])
        }
        return out_doc

    def doc_to_text(self, doc):
        return "{}".format(doc["title"])

    def doc_to_target(self, doc):
        return " {}".format({0: "혐오", 1: "악플", 2: "양호"}[doc["gold"]])

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]
        return {
            "f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "f1": True
        }

    def aggregation(self):
        return {
            "f1": macro_f1_score
        }
