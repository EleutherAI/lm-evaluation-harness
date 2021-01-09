# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

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

    # TODO: Implement evaluation code

    # ***IMPORTANT***: this evaluation function needs to be written for the new framework. 
    # For more info, check out the interface in base.py and the example BoolQ implementation in superglue.py. 
    # Remove this comment when the evaluation code is implemented.


class WikiText2(NLP_TASK):
    NLP_PATH = "wikitext"
    NLP_NAME = "wikitext-2-raw-v1"

    def fewshot_description(self):
        return ""

    def doc_to_text(self, doc, include_target=True):
        return doc['text']

    # TODO: Implement evaluation code

    # ***IMPORTANT***: this evaluation function needs to be written for the new framework. 
    # For more info, check out the interface in base.py and the example BoolQ implementation in superglue.py. 
    # Remove this comment when the evaluation code is implemented.