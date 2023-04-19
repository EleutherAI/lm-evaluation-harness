"""
XNLI: Evaluating Cross-lingual Sentence Representations
https://arxiv.org/abs/1809.05053

@misc{conneau2018xnli,
      title={XNLI: Evaluating Cross-lingual Sentence Representations},
      author={Alexis Conneau and Guillaume Lample and Ruty Rinott and Adina Williams and Samuel R. Bowman and Holger Schwenk and Veselin Stoyanov},
      year={2018},
      eprint={1809.05053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


class XNLIBase(Task):
    VERSION = 0
    DATASET_PATH = "xnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return self.dataset["test"]

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class XNLIDe(XNLIBase):
    DATASET_NAME = "de"  # German part of xnli

    def doc_to_text(self, doc):
        return (
            "Prämisse: {}\nHypothese: {} Wahr, Falsch oder Neutral?\nAntwort:".format(
                doc["premise"],
                doc["hypothesis"].strip()
                + ("" if doc["hypothesis"].strip().endswith(".") else "."),
            )
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "Wahr", 1: "Neutral", 2: "Falsch"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " Wahr")
        ll_neither, _ = rf.loglikelihood(ctx, " Neutral")
        ll_false, _ = rf.loglikelihood(ctx, " Falsch")
        return ll_true, ll_neither, ll_false


class XNLIEn(XNLIBase):
    DATASET_NAME = "en"

    def doc_to_text(self, doc):
        return (
            # "Premise: {}\nHypothesis: {} True, False or Neither?\nAntwort:".format(
            "Premise: {}\nHypothesis: {} True, False or Neither?\nAnswer:".format(
                doc["premise"],
                doc["hypothesis"].strip()
                + ("" if doc["hypothesis"].strip().endswith(".") else "."),
            )
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false


class XNLIFr(XNLIBase):
    DATASET_NAME = "fr"

    def doc_to_text(self, doc):
        return "Prémisse: {}\nHypothèse: {} Vrai, Faux or Neutre?\nRéponse:".format(
            doc["premise"],
            doc["hypothesis"].strip()
            + ("" if doc["hypothesis"].strip().endswith(".") else "."),
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "Vrai", 1: "Neutre", 2: "Faux"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " Vrai")
        ll_neither, _ = rf.loglikelihood(ctx, " Neutre")
        ll_false, _ = rf.loglikelihood(ctx, " Faux")
        return ll_true, ll_neither, ll_false


class XNLIEs(XNLIBase):
    DATASET_NAME = "es"

    def doc_to_text(self, doc):
        return (
            "Premisa: {}\nHipótesis: {} Verdadero, Falso o Neutro?\nRespuesta:".format(
                doc["premise"],
                doc["hypothesis"].strip()
                + ("" if doc["hypothesis"].strip().endswith(".") else "."),
            )
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "Verdadero", 1: "Neutro", 2: "Falso"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " Verdadero")
        ll_neither, _ = rf.loglikelihood(ctx, " Neutro")
        ll_false, _ = rf.loglikelihood(ctx, " Falso")
        return ll_true, ll_neither, ll_false
