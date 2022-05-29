"""
KOBEST
https://arxiv.org/abs/2204.04541

A well-formulated benchmark plays a critical role in spurring advancements 
in the natural language processing (NLP) field, as it allows objective and
precise evaluation of diverse models. As modern language models (LMs) have 
become more elaborate and sophisticated, more difficult benchmarks that require
linguistic knowledge and reasoning have been proposed. However, most of these
benchmarks only support English, and great effort is necessary to construct
benchmarks for other low resource languages. To this end, we propose a new
benchmark named Korean balanced evaluation of significant tasks (KoBEST),
which consists of five Korean-language downstream tasks. Professional Korean
linguists designed the tasks that require advanced Korean linguistic knowledge.
Moreover, our data is purely annotated by humans and thoroughly reviewed to
guarantee high data quality. We also provide baseline models and human performance
results. Our dataset is available on the Huggingface.

Homepage: https://huggingface.co/datasets/skt/kobest_v1
"""

import numpy as np
from lm_eval.base import MultipleChoiceTask, rf, Task
from lm_eval.metrics import f1_score, macro_f1_score


class BoolQ(Task):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "boolq"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "{} 질문: {} 답변: ".format(doc["paragraph"], doc["question"])

    def doc_to_target(self, doc):
        return " {}".format({0: "아니오.", 1: "예."})
        
    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 아니오.")
        ll_yes, _ = rf.loglikelihood(ctx, " 예.")

        return ll_no, ll_yes

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
            "f1": f1_score
        }

class WiC(Task):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "wic"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "문장1: {} 문장2: {} 두 문장에서 {}가 같은 뜻으로 쓰였나?".format(doc["context_1"], doc["context_2"], doc["word"])

    def doc_to_target(self, doc):
        return " {}".format({0: "아니오", 1: "예"})
        
    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 아니오")
        ll_yes, _ = rf.loglikelihood(ctx, " 예")

        return ll_no, ll_yes

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
            "f1": f1_score
        }


class SentiNeg(Task):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "sentineg"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "문장: {} 긍부정:".format(doc["sentence"])

    def doc_to_target(self, doc):
        return " {}".format({0: "부정", 1: "긍정"})
        
    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 부정")
        ll_yes, _ = rf.loglikelihood(ctx, " 긍정")

        return ll_no, ll_yes

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
            "f1": f1_score
        }
