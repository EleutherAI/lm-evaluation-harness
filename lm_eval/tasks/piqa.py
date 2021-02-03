import json
import random
from lm_eval.base import Task, rf, mean
from ..utils import sh
import os

class PiQA(Task):
    def download(self):
        if not os.path.exists('data/piqa'):
            #TODO: use best_download
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
        return False

    def load_docs(self, textfilename, labelfilename):
        if labelfilename != None:
            return zip([json.loads(entry) for entry in list(open(textfilename,'r'))],list(map(lambda x: x.strip(), open(labelfilename, 'r'))))
        else:
            return [json.loads(entry) for entry in list(open(textfilename,'r'))]
    
    def training_docs(self):
        return self.load_docs('data/piqa/piqa-train.jsonl', 'data/piqa/piqa-train-labels.lst')
   
    def validation_docs(self):
        return self.load_docs('data/piqa/piqa-valid.jsonl', 'data/piqa/piqa-valid-labels.lst')

    #def test_docs(self):
    #    return self.load_docs('data/piqa/piqa-test.jsonl', None)
    
    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""
    
    def doc_to_text(self, doc):
        return doc[0]['goal']

    def doc_to_target(self, doc):
        #TODO: check if oa uses newline
        rightanswer = int(doc[1]) + 1
        return '\n' + ''.join([doc[0]['goal'],' ',doc[0]['sol'+str(rightanswer)]])

    def construct_requests(self, doc, ctx):
        ll_1, _ = rf.loglikelihood(ctx, doc[0]['sol1'])
        ll_2, _ = rf.loglikelihood(ctx, doc[0]['sol2'])

        return ll_1, ll_2
    
    def process_results(self, doc, results):
        ll_1, ll_2 = results

        return {
            'acc': (ll_1 > ll_2) == (int(doc[1]) == 0)
        }

    def aggregation(self):
        return {
            'acc': mean
        }

    def higher_is_better(self):
        return {
            'acc': True
        }
