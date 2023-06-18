"""
Korean legal AI datasets, LBox OPEN
Multi-task on Legal corpus 
https://arxiv.org/pdf/2206.05224.pdf
"""

import numpy as np
from lm_eval.base import Task, MultipleChoiceTask, rf
from lm_eval.metrics import macro_f1_score, mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize

_CITATION ="""
@article{hwang2022multi,
  title={A multi-task benchmark for korean legal language understanding and judgement prediction},
  author={Hwang, Wonseok and Lee, Dongjun and Cho, Kyoungyeon and Lee, Hanuhl and Seo, Minjoon},
  journal={arXiv preprint arXiv:2206.05224},
  year={2022}
}
"""

class LegalCasename(Task):
    VERSION = 0
    DATASET_PATH = "lbox/lbox_open"
    DATASET_NAME = "casename_classification"
    
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
        return map(self._process_doc, self.dataset["valid"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return "문장: {} ".format(doc["facts"])

    def doc_to_target(self, doc):
        return " {}".format({"civil": "민사", "criminal": "형사"}[doc["casetype"]])

    def construct_requests(self, doc, ctx):
        ll_m, _ = rf.loglikelihood(ctx, " 민사")
        ll_h, _ = rf.loglikelihood(ctx, " 형사")
        return ll_m, ll_h
    
    def process_results(self, doc, results):
        ll_m, ll_h = results
        pred = ll_h > ll_m
        gold = {"civil": 0, "criminal": 1}[doc["casetype"]]
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

