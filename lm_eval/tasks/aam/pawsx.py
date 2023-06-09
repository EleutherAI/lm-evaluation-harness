"""
PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification
https://arxiv.org/abs/1908.11828

@misc{yang2019pawsx,
      title={PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification},
      author={Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
      year={2019},
      eprint={1908.11828},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

from lm_eval.base import rf, Task
from lm_eval.metrics import mean, f1_score
from lm_eval.utils import general_detokenize


class PAWSXBase(Task):
    VERSION = 0
    DATASET_PATH = "paws-x"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class PAWSXDe(PAWSXBase):
    DATASET_NAME = "de"  # German part of paws-x

    def fewshot_description(self):
        return "Entscheide, ob beide Sätze dieselbe Bedeutung haben."

    def doc_to_text(self, doc):
        return "Satz 1: {}\nSatz 2: {}\nFrage: Haben beide Sätze die gleich Bedeutung?\nAntwort:".format(
            general_detokenize(doc["sentence1"]),
            general_detokenize(doc["sentence2"]),
        )

    def doc_to_target(self, doc):
        return " {}".format("Ja" if (doc["label"]) else "Nein")

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " Ja")
        ll_no, _ = rf.loglikelihood(ctx, " Nein")
        return ll_yes, ll_no
