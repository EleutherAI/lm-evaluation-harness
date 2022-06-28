"""
https://arxiv.org/abs/1909.01066
https://arxiv.org/abs/2005.04611
LAMA is a prob dataset to test the factual and commonsense knowledge in language models. The dataset includes a subset of
Google_RE (https://code.google.com/archive/p/relation-extraction-corpus/), TRex (subset of wikidata triples),
Conceptnet (https://github.com/commonsense/conceptnet5/wiki) and Squad.

Homepage: https://github.com/facebookresearch/LAMA
"""
from typing import Optional

from lm_eval.api.task import PromptSourceTask
from lm_eval.api.metric import mean


_CITATION = """
@inproceedings{petroni2019language, title={Language Models as Knowledge Bases?},
               author={F. Petroni, T. Rockt{"{a}}schel, A. H. Miller, P. Lewis, A. Bakhtin, Y. Wu and S. Riedel},
               booktitle={In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019}, year={2019} }

@inproceedings{petroni2020how,
               title={How Context Affects Language Models' Factual Predictions},
               author={Fabio Petroni and Patrick Lewis and Aleksandra Piktus and Tim Rockt{"a}schel and Yuxiang Wu and Alexander H. Miller and Sebastian Riedel},
               booktitle={Automated Knowledge Base Construction}, year={2020}, url={https://openreview.net/forum?id=025X0zPfn} }
"""


class BigScienceLAMA(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "janck/bigscience-lama"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]


class Trex(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "lama"
    DATASET_NAME = "trex"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["train"]

    def process_results(self, doc, results):
        out = {}
        # gold = doc
        pred = results[0].strip()
        target = self.doc_to_target(doc)["obj_label"]
        out["acc"] = pred == target

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
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["train"]

    def process_results(self, doc, results):
        out = {}
        pred = results[0].strip()

        target = self.doc_to_target(doc)["obj_label"]
        out["acc"] = pred == target

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
        return False

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return False

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["train"]

    def process_results(self, doc, results):
        out = {}
        pred = results[0].strip()

        target = self.doc_to_target(doc)["obj_label"]
        out["acc"] = pred == target

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
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["train"]

    def process_results(self, doc, results):
        out = {}
        pred = results[0].strip()
        target = self.doc_to_target(doc)["obj_label"]
        out["acc"] = pred == target

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
