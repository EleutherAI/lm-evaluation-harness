import numpy as np
from lm_eval.base import rf, mean
from . common import HFTask


class PiQA(HFTask):
    DATASET_PATH = "piqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def doc_to_text(self, doc):
        return "Question: "+doc["goal"] + "\nAnswer:"

    def doc_to_target(self, doc):
        solutions = [doc["sol1"], doc["sol2"]]
        return " " + solutions[doc["label"]]

    def construct_requests(self, doc, ctx):
        ll_1, _ = rf.loglikelihood(ctx, " " + doc['sol1'])
        ll_2, _ = rf.loglikelihood(ctx, " " + doc['sol2'])
        return ll_1, ll_2

    def process_results(self, doc, results):
        return {
            'acc': np.argmax(results) == doc["label"]
        }

    def aggregation(self):
        return {
            'acc': mean
        }

    def higher_is_better(self):
        return {
            'acc': True
        }
