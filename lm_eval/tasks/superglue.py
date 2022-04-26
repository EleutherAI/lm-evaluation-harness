"""
SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems
https://w4ngatang.github.io/static/papers/superglue.pdf

SuperGLUE is a benchmark styled after GLUE with a new set of more difficult language
understanding tasks.

Homepage: https://super.gluebenchmark.com/

TODO: WSC requires free-form generation.
"""
import numpy as np
import sklearn
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.base import rf, PromptSourceTask
from lm_eval.metrics import mean, acc_all, metric_max_over_ground_truths, yesno
from lm_eval.utils import general_detokenize


_CITATION = """
@inproceedings{NEURIPS2019_4496bf24,
    author = {Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {},
    publisher = {Curran Associates, Inc.},
    title = {SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
    url = {https://proceedings.neurips.cc/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf},
    volume = {32},
    year = {2019}
}
"""


class BoolQ(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "super_glue"
    DATASET_NAME = "boolq"

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
        return self.dataset["validation"]


class CommitmentBank(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "super_glue"
    DATASET_NAME = "cb"

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
        return self.dataset["validation"]

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc, "f1": (pred, gold)}

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    @classmethod
    def cb_multi_fi(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
        f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
        f13 = sklearn.metrics.f1_score(y_true=golds == 2, y_pred=preds == 2)
        avg_f1 = mean([f11, f12, f13])
        return avg_f1

    def aggregation(self):
        return {
            "acc": mean,
            "f1": self.cb_multi_fi,
        }


class Copa(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "copa"

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
        return self.dataset["validation"]

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]


class MultiRC(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "super_glue"
    DATASET_NAME = "multirc"

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
        return self.dataset["validation"]

    def process_results(self, doc, results):
        ll_true_choice, ll_false_choice = results
        pred = ll_true_choice > ll_false_choice
        return {"acc": (pred, doc)}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": acc_all}


class ReCoRD(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "record"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        # In ReCoRD, each doc manifests multiple "examples" in the context of few shot example packing.
        # Each doc consists of multiple answer candidates, each of which is scored yes/no.
        if self._training_docs is None:
            self._training_docs = []
            for doc in self.dataset["train"]:
                self._training_docs.append(self._process_doc(doc))
        return self._training_docs

    def validation_docs(self):
        # See: training_docs
        for doc in self.dataset["validation"]:
            yield self._process_doc(doc)

    @classmethod
    def _process_doc(cls, doc):
        return {
            "passage": doc["passage"],
            "query": doc["query"],
            "entities": sorted(list(set(doc["entities"]))),
            "answers": sorted(list(set(doc["answers"]))),
        }

    def process_results(self, doc, results):
        # ReCoRD's evaluation is actually deceptively simple:
        # - Pick the maximum likelihood prediction entity
        # - Evaluate the accuracy and token F1 PER EXAMPLE
        # - Average over all examples

        # TODO (jon-tow): Look at result
        max_idx = np.argmax(np.array([result[0] for result in results]))

        prediction = doc["entities"][max_idx]
        gold_label_set = doc["answers"]
        f1 = metric_max_over_ground_truths(
            squad_metrics.compute_f1, prediction, gold_label_set
        )
        em = metric_max_over_ground_truths(
            squad_metrics.compute_exact, prediction, gold_label_set
        )

        return {
            "f1": f1,
            "em": em,
        }

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }


class WordsInContext(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wic"

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
        return self.dataset["validation"]

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class SGWinogradSchemaChallenge(PromptSourceTask):
    VERSION = 0
    # Note: This implementation differs from Fig G.32 because this is the SuperGLUE,
    #       binary version of the task.
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wsc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # GPT-3 Paper's format only uses positive examples for fewshot "training"
                self._training_docs = [
                    doc for doc in self.dataset["train"] if doc["label"]
                ]
            return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
