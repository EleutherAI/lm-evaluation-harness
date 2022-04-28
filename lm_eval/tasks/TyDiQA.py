"""
Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference
https://arxiv.org/abs/1902.01007

A controlled evaluation set called HANS (Heuristic Analysis for NLI Systems),
which contains many examples where the heuristics fail.

Homepage: https://github.com/tommccoy1/hans
"""
from lm_eval.base import PromptSourceTask


_CITATION = """\
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
"""


class Primary(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "tydiqa"
    DATASET_NAME = "primary_task"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
    def process_results(self, doc, results):
        out = {}
        #gold = doc
        pred = results[0].strip()
        print("############")
        print(self.doc_to_target(doc))

        target = self.doc_to_target(doc)['sub_label']
        #pred = np.argmax(results)
        out["acc"] = pred == target


        #result = metric.compute(predictions=pred, references=gold)
        #out['acc'] = {"accuracy": result["score"]}
        
        #out['acc'] = 1.0 if pred == gold else 0.0
        if self.save_examples:
            example = {
                "pred": pred,
                "target": target,
            }
            return out, example

        return out


class Secondary(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "tydiqa"
    DATASET_NAME = "secondary_task"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


