import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from . common import HFTask, simple_accuracy_metric, yesno

class Winogrande(HFTask):
    DATASET_PATH = "winogrande"
    DATASET_NAME = "winogrande_xl"

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
        return "Winograd schema sentence including a either a ___ blank with a missing word, making the pronoun ambiguous, or the same with the word filled in."

    def doc_to_text(self, doc, include_target=True):
        text = doc['sentence']
        if include_target:
            answer_n = doc['answer']
            if answer_n == '1':
                answer = doc['option1']
            elif answer_n == '2':
                answer = doc['option2']
            else:
                raise ValueError("Winogrande from HF datasets contained an invalid answer key")
            text = text.replace("_", answer)
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: Write evaluation function
        raise NotImplementedError()