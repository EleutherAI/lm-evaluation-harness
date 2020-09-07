import abc
import nlp
import numpy as np
from ..base import Dataset


class NLP_TASK(Dataset):
    NLP_PATH = None
    NLP_NAME = None

    def _load_nlp_dataset(self):
        return nlp.load_dataset(path=self.NLP_PATH, name=self.NLP_NAME)

    def training_docs(self):
        if self.has_training_docs():
            return self._load_nlp_dataset()["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self._load_nlp_dataset()["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self._load_nlp_dataset()["test"]


def simple_accuracy_metric(preds, golds):
    acc = float((np.array(preds) == np.array(golds)).mean())
    return {
        "major": acc,
        "minor": {"acc": acc},
        "higher_is_better": True,
    }


def yesno(x):
    if x:
        return 'yes'
    else:
        return 'no'
