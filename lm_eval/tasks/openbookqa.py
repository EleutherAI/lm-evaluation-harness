# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from . common import HFTask, simple_accuracy_metric, yesno

class OpenBookQA(HFTask):
    DATASET_PATH = "openbookqa"
    DATASET_NAME = "main"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.data["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test"]

    def fewshot_description(self):
        return "Text of the question prompt\nText of the answer completion"

    def doc_to_text(self, doc, include_target=True):
        text = doc['question_stem'] + '\n'
        if include_target:
            letter_answer = doc['answerKey']
            if letter_answer == 'A':
                index = 0
            elif letter_answer == 'B':
                index = 1
            elif letter_answer == 'C':
                index = 2
            elif letter_answer == 'D':
                index = 3
            else:
                raise ValueError("OpenBookQA from HF datasets contained an invalid answer key")
            text += doc['choices']['text'][index] + '.'
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: Write evaluation function
        raise NotImplementedError()
