import numpy as np
import json
import random
from .common import HFTask 
from lm_eval.base import rf, mean


class Pubmed_QA(HFTask):
    DATASET_PATH = "pubmed_qa"
    DATASET_NAME = "pqa_labeled"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.has_test_docs():
            # HF is labelled as train but its really just for testing
            return self.data["train"]

    def fewshot_description(self):
        # Average ctx length in labelled dataset is 238.9
        # 2 few-shot exmamples pushes it beyond context window
        return ""

    def doc_to_text(self, doc):
        ctxs = "\n".join(doc["context"]["contexts"])
        return "Abstract: {}\nQuestion: {}\nAnswer:".format(
            ctxs,
            doc["question"],
            doc["final_decision"]
        )

    def doc_to_target(self, doc):
        return " {}".format(doc["final_decision"])

    def fewshot_examples(self, k):
        # Since only test docs sample from test docs
        if self._training_docs is None:
            self._training_docs = list(self.test_docs())
        return random.sample(self._training_docs, k)

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns
        an iterable of Requests which will be sent to the LM.
        """
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        ll_maybe, _ = rf.loglikelihood(ctx, " maybe")
        return ll_yes, ll_no, ll_maybe

    def process_results(self, doc, results):
        gold = doc["final_decision"]
        ll_yes, ll_no, ll_maybe = results
        pred = np.argmax(results)
        return {
            "acc": ["yes", "no", "maybe"][pred] == gold, 
        }

    def aggregation(self):
        return {
            "acc" : mean
        }

    def higher_is_better(self):
        return {
            "acc" : True
        }
