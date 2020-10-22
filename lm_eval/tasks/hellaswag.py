import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from . common import HFTask, simple_accuracy_metric, yesno

class HellaSwag(HFTask):
    DATASET_PATH = "hellaswag"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.data["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test"]

    def fewshot_description(self):
        return "Label for the relevant action: Sentences describing the context, with an incomplete sentence trailing\nanswer that plausibly completes the situation."

    def doc_to_text(self, doc, include_target=True):
        text = doc['activity_label'] + ': ' + doc['ctx'] + '\n'
        if include_target:
            letter_answer = doc['label']
            if letter_answer == '0':
                index = 0
            elif letter_answer == '1':
                index = 1
            elif letter_answer == '2':
                index = 2
            elif letter_answer == '3':
                index = 3
            else:
                raise ValueError("HellaSwag from HF datasets contained an invalid answer key")
            text += doc['endings'][index]
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: Write evaluation function
        raise NotImplementedError()
