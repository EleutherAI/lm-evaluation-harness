"""
NSMC:
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize

_CITATION = """
@inproceedings{zellers2019hellaswag,
    title={NSMC: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
"""


class NSMC(Task):
    VERSION = 0
    DATASET_PATH = "nsmc"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "다음 문장은 긍정일까요 부정일까요?\n{}\n정답:".format(
            general_detokenize(doc["document"]),
        )

    def doc_to_target(self, doc):
        return " {}".format({1: "긍정", 0: "부정"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, " 긍정")
        ll_negative, _ = rf.loglikelihood(ctx, " 부정")
        return ll_positive, ll_negative

    def process_results(self, doc, results):
        ll_positive, ll_negative = results
        pred = ll_positive > ll_negative
        gold = doc["label"]
        return {
            "acc": pred == gold
        }

    def higher_is_better(self):
        return {
            "acc": True
        }

    def aggregation(self):
        return {
            "acc": mean
        }