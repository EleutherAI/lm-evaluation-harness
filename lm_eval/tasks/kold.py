"""
KOLD
https://arxiv.org/abs/2205.11315

Homepage: https://github.com/boychaboy/KOLD 
"""
import inspect
import lm_eval.datasets.kold.kold
import numpy as np
from lm_eval.base import MultipleChoiceTask, rf, Task
from lm_eval.metrics import macro_f1_score, mean, f1_score

def make_label(label: str):
    """
    0 = 없음
    1 = 개인
    2 = 단체
    """
    if label == "individual":
        return 1
    elif label == "group":
        return 2
    else:
        return 0

class KoldLevelA(Task):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.kold.kold)
    DATASET_NAME = "kolda"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc):
            prompt = f"문장: {doc['comment']}\n"
            prompt += f"질문: 문장이 공격적인가요?\n정답:"
            return prompt
        
        return {
            "prompt": format_example(doc),
            "label": doc["off"]
        }


    def doc_to_text(self, doc):
        return doc["prompt"]

    def doc_to_target(self, doc):
        return " {}".format({0: "아니오", 1: "예"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 아니오")
        ll_yes, _ = rf.loglikelihood(ctx, " 예")

        return ll_no, ll_yes

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {
            "acc": pred == gold,
            "f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "f1": f1_score
        }



class KoldLevelB(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.kold.kold)
    DATASET_NAME = "koldb"


    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, choices):
            prompt = f"문장: {doc['comment']}\n"
            prompt += "질문: 공격 대상이 "
            prompt += "".join([f"{choice} "for choice in choices])
            prompt += "중 무엇인가요?\n정답:"
            return prompt

        choices = ["없음", "개인", "단체"]
        return {
            "prompt": format_example(doc, choices),
            "choices": choices,
            "label": make_label(doc["tgt"])
        }


    def doc_to_text(self, doc):
        return doc["prompt"]

    def doc_to_target(self, doc):
        return " {}".format({0: "없음", 1: "개인", 2:"단체"}[doc["label"]])

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
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
