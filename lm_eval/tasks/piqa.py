# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

import json
import random
from lm_eval.base import Dataset
from ..utils import sh

class PiQA(Dataset):
    def __init__(self):
        self.download()
    def download(self):
        #pass
        #TODO: don't download if files already there
        sh("""
           mkdir -p data/piqa
           wget https://yonatanbisk.com/piqa/data/train.jsonl -O data/piqa/piqa-train.jsonl
           wget https://yonatanbisk.com/piqa/data/train-labels.lst -O data/piqa/piqa-train-labels.lst
           wget https://yonatanbisk.com/piqa/data/valid.jsonl -O data/piqa/piqa-valid.jsonl
           wget https://yonatanbisk.com/piqa/data/valid-labels.lst -O data/piqa/piqa-valid-labels.lst
           wget https://yonatanbisk.com/piqa/data/tests.jsonl -O data/piqa/piqa-test.jsonl
           """)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def load_docs(self, textfilename, labelfilename):
        if labelfilename != None:
            return zip([json.loads(entry) for entry in list(open(textfilename,'r'))],list(open(labelfilename, 'r')))
        else:
            return [json.loads(entry) for entry in list(open(textfilename,'r'))]
    
    def training_docs(self):
        return self.load_docs('data/piqa/piqa-train.jsonl', 'data/piqa/piqa-train-labels.lst')
   
    def validation_docs(self):
        return self.load_docs('data/piqa/piqa-valid.jsonl', 'data/piqa/piqa-valid-labels.lst')

    def test_docs(self):
        return self.load_docs('data/piqa/piqa-test.jsonl', None)
    
    def fewshot_description(self):
        pass
    
    def doc_to_text(self, doc, include_target=True):
        if include_target:
            rightanswer = int(doc[1][0])+1
            return ''.join([doc[0]['goal'],' ',doc[0]['sol'+str(rightanswer)]])
        #TODO: check if oa uses newline
        return  doc['goal'] + ' '

    def evaluate(self, docs, lm):
        pass

