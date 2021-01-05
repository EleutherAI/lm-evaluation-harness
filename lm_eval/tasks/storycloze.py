# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

import json
import random
from lm_eval.base import Dataset
from ..utils import sh
import csv

class StoryCloze(Dataset):
    def __init__(self):
        self.download()
    def download(self):
        #TODO: replace with Eye link
        pass

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def load_doc(self, filename):
        with open(filename, newline='') as file:
            filereader = csv.reader(file)
            return list(filereader)
                

    def validation_docs(self):
        return  self.load_doc("data/storycloze/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv")

    def test_docs(self):
        return self.load_doc("data/storycloze/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv")

    
    def fewshot_description(self):
        pass
    
    def doc_to_text(self, doc, include_target=True):
        if include_target:
            return ' '.join([*doc[1:5],doc[int(doc[-1])-4]]) 
        else:
            return ' '.join([*doc[1:5]])

    def evaluate(self, docs, lm):
        pass

