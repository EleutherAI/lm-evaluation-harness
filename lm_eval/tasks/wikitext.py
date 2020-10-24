import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from . common import NLP_TASK, simple_accuracy_metric, yesno

class WikiText103(NLP_TASK):
    NLP_PATH = "wikitext"
    NLP_NAME = "wikitext-103-raw-v1"

    def fewshot_description(self):
        return ""

    def doc_to_text(self, doc, include_target=True):
        return doc['text']
    def evaluate(self, docs, lm, provide_description, num_fewshot):
        pass


class WikiText2(NLP_TASK):
    NLP_PATH = "wikitext"
    NLP_NAME = "wikitext-2-raw-v1"

    def fewshot_description(self):
        return ""

    def doc_to_text(self, doc, include_target=True):
        return doc['text']
    def evaluate(self, docs, lm, provide_description, num_fewshot):
        pass