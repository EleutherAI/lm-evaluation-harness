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
            "choices": ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오", "악플/욕설", "clean"],
            "gold": np.argmax(doc["labels"])
        }
        return out_doc

    def doc_to_text(self, doc):
        return "{}".format(doc["title"])

    def doc_to_target(self, doc):
        return " {}".format({0: "여성/가족", 1: "남성", 2: "성소수자", 3: "인종/국적", 4: "연령", 5: "지역", 6: "종교", 7: "기타 혐오", 8: "악플/욕설", 9: "clean"}[doc["gold"]])

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
