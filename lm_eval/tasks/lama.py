"""
Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference
https://arxiv.org/abs/1902.01007

A controlled evaluation set called HANS (Heuristic Analysis for NLI Systems),
which contains many examples where the heuristics fail.

Homepage: https://github.com/tommccoy1/hans
"""
from lm_eval.base import PromptSourceTask
import numpy as np 
from lm_eval.metrics import mean
from lm_eval import metrics,utils
from typing import Iterable, Optional

_CITATION = """
@inproceedings{petroni2019language, title={Language Models as Knowledge Bases?},
               author={F. Petroni, T. Rockt{"{a}}schel, A. H. Miller, P. Lewis, A. Bakhtin, Y. Wu and S. Riedel},
               booktitle={In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019}, year={2019} }

@inproceedings{petroni2020how,
               title={How Context Affects Language Models' Factual Predictions},
               author={Fabio Petroni and Patrick Lewis and Aleksandra Piktus and Tim Rockt{"a}schel and Yuxiang Wu and Alexander H. Miller and Sebastian Riedel},
               booktitle={Automated Knowledge Base Construction}, year={2020}, url={https://openreview.net/forum?id=025X0zPfn} }
"""

class Trex(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lama"
    DATASET_NAME = "trex"

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return False

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["train"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def process_results(self, doc, results):
        out = {}
        #gold = doc
        pred = results[0].strip()
        target = self.doc_to_target(doc)['obj_label']
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

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    def doc_to_target(self, doc):
        return doc


class google_re(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lama"
    DATASET_NAME = "google_re"

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return False

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["train"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def process_results(self, doc, results):
        out = {}
        #gold = doc
        pred = results[0].strip()

        target = self.doc_to_target(doc)['obj_label']
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

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    def doc_to_target(self, doc):
        return doc

class Conceptnet(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lama"
    DATASET_NAME = "conceptnet"

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return False

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["train"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def process_results(self, doc, results):
        out = {}
        #gold = doc
        pred = results[0].strip()

        target = self.doc_to_target(doc)['obj_label']
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

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    def doc_to_target(self, doc):
        return doc


class Squad(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lama"
    DATASET_NAME = "squad"

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return False

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["train"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def process_results(self, doc, results):
        out = {}
        #gold = doc
        pred = results[0].strip()
        print("################")
        print(pred)
        target = self.doc_to_target(doc)['obj_label']
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

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    def doc_to_target(self, doc):
        return doc

    def max_generation_length(self) -> Optional[int]:
        """Denote where the max length of the generation if it is obvious from the task."""
        return 5

