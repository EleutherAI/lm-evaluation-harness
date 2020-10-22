import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from . common import HFTask, simple_accuracy_metric, yesno

class SQuAD(HFTask):
    DATASET_PATH = "squad_v2"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            return self.data["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation"]

    def fewshot_description(self):
        return "Title: The_Title_of_It\n\nBackground: A text passage as background to answer the question with.\n\nQ: Question about the passage.\n\nA: Answer."

    def doc_to_text(self, doc, include_target=True):
        text = 'Title: ' + doc['title'] + '\n\n' + 'Background: ' + doc['context'] + '\n\n' + 'Q: ' + doc['question'] + '\n\n' + 'A: '
        if include_target:
            answer_list = doc['answers']['text']
            if len(answer_list) > 0:
                answer = answer_list[0]
            else:
                answer = 'unanswerable'
            text += answer
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: Write evaluation function
        raise NotImplementedError()