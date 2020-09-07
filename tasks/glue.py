import nlp
import numpy as np
import random
from sklearn.metrics import f1_score, matthews_corrcoef
from . common import NLP_TASK, simple_accuracy_metric
from . import TASK_REGISTRY


@TASK_REGISTRY.register("cola")
class CoLA(NLP_TASK):
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "Does this sentence make sense?:\tTrue or False?" \
               "\nsentence:{}\nAnswer: ".format(doc["sentence"])
        if include_target:
            text += " {}".format({1: "True", 0: "False"}[doc["label"]])
        return text

    def evaluate(self, docs, lm, k=0):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in docs:
            word = lm.generate(
                context=self.fewshot_context(doc=doc, k=k),
                max_gen_length=1,
            )
            if word.strip() == "True":
                preds.append(1)
            elif word.strip() == "False":
                preds.append(0)
            else:
                preds.append(-1)
        golds = np.array(golds)
        preds = np.array(preds)
        mcc = float(matthews_corrcoef(y_true=golds, y_pred=preds))
        return {
            "major": mcc,
            "minor": {"mcc": mcc},
            "higher_is_better": True,
        }


@TASK_REGISTRY.register("mnli")
class MNLI(NLP_TASK):
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        if self.has_validation_docs():
            return self._load_nlp_dataset()["validation_matched"]

    def test_docs(self):
        if self.has_test_docs():
            return self._load_nlp_dataset()["test_matched"]

    def doc_to_text(self, doc, include_target=True):
        text = "{}\nquestion:\t{}\tTrue, False or Neither?\nanswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )
        if include_target:
            # True = entailment
            # False = contradiction
            # Neither = neutral
            text += " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])
        return text

    def evaluate(self, docs, lm, k=0):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in docs:
            word = lm.generate(
                context=self.fewshot_context(doc=doc, k=k),
                max_gen_length=1,
            )
            if word.strip() == "True":
                preds.append(1)
            elif word.strip() == "False":
                preds.append(0)
            else:
                preds.append(-1)
        return simple_accuracy_metric(preds=preds, golds=golds)


@TASK_REGISTRY.register("rte")
class RTE(NLP_TASK):

    NLP_PATH = "glue"
    NLP_NAME = "rte"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "{}\nquestion:\t{}\tTrue or False?\nanswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )
        if include_target:
            text += " {}".format({1: "True", 0: "False"}[doc["label"]])
        return text

    def evaluate(self, docs, lm, k=0):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in docs:
            word = lm.generate(
                context=self.fewshot_context(doc=doc, k=k),
                max_gen_length=1,
            )
            if word.strip() == "True":
                preds.append(1)
            elif word.strip() == "False":
                preds.append(0)
            else:
                preds.append(-1)
        return simple_accuracy_metric(preds=preds, golds=golds)
