"""
KLUE
https://arxiv.org/abs/2105.09680

 Korean Language Understanding Evaluation (KLUE) benchmark is a series of datasets
 to evaluate natural language understanding capability of Korean language models.
 KLUE consists of 8 diverse and representative tasks, which are accessible to anyone without any restrictions.
 With ethical considerations in mind, we deliberately design annotation guidelines
 to obtain unambiguous annotations for all datasets. Furthermore, we build an evaluation system
 and carefully choose evaluations metrics for every task, thus establishing fair comparison across Korean language models.
 
 Homepage: https://klue-benchmark.com/
"""

import numpy as np
from lm_eval.base import MultipleChoiceTask, rf
from lm_eval.metrics import macro_f1_score


class YNAT(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "ynat"

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
        return map(self._process_doc,self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "title": doc["title"],
            "choices": ["과학", "경제", "사회", "생활", "세계", "스포츠", "정치"],
			"gold": doc["label"]
        }
        return out_doc

    def doc_to_text(self, doc):
        return "다음 문장의 카테고리는?\n{}\n답변:".format(doc["title"])

    def doc_to_target(self, doc):
        return " {}".format({0: "과학", 1: "경제", 2: "사회", 3: "생활", 4: "세계", 5: "스포츠", 6: "정치"}[doc["gold"]])

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
