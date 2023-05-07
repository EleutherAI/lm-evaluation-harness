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
from lm_eval.metrics import macro_f1_score, mean


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
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }


class COPA(Task):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "copa"

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
        '''
        Connector: “왜냐하면” if Question is “원인” else “그래서”
        Format: “{Premise} {Connector} {Answer Alternative}”
        '''
        connector = {
            "원인": "왜냐하면",
            "결과": "그래서",
        }[doc["question"].strip()]

        return doc["premise"] + f" {connector}"

    def doc_to_target(self, doc):
        correct_choice = doc["alternative_1"] if doc["label"] == 0 else doc["alternative_2"]

        return " " + correct_choice
        
    def construct_requests(self, doc, ctx):
        ll_choice1, _ = rf.loglikelihood(ctx, " "+doc["alternative_1"])
        ll_choice2, _ = rf.loglikelihood(ctx, " "+doc["alternative_2"])

        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }


    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
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
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }


class HellaSwag(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "skt/kobest_v1"
    DATASET_NAME = "hellaswag"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "문장: {}".format(doc["context"]),
            "choices": [doc["ending_1"], doc["ending_2"], doc["ending_3"], doc["ending_4"]],
            "gold": int(doc['label']),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]

        acc = 1. if np.argmax(results) == gold else 0.
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1. if np.argmax(results / completion_len) == gold else 0.

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "macro_f1": macro_f1_score
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
        return " {}".format({0: "부정", 1: "긍정"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 부정")
        ll_yes, _ = rf.loglikelihood(ctx, " 긍정")

        return ll_no, ll_yes

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }
