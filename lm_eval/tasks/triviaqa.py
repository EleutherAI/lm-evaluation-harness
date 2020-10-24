import json
import random
from lm_eval.base import Dataset
from ..utils import sh

class TriviaQA(Dataset):
    def __init__(self):
        self.download()
    def download(self):
        pass
        #TODO: don't download if files already there
        sh("""
           mkdir -p data/triviaqa
           wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz -O data/triviaqa/trivia_qa-unfiltered.tar.gz
           tar -xf data/triviaqa/trivia_qa-unfiltered.tar.gz
           mv triviaqa-unfiltered/ data/triviaqa/
           """)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return json.load(open('data/triviaqa/triviaqa-unfiltered/unfiltered-web-train.json'))['Data']

    def validation_docs(self):
        return  json.load(open('data/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json'))['Data']

    def test_docs(self):
        return  json.load(open('data/triviaqa/triviaqa-unfiltered/unfiltered-web-test.json'))['Data']     
    
    def fewshot_description(self):
        pass
    
    def doc_to_text(self, doc, include_target=True):
        return ''.join(['Q: ', doc['Question'], '\n\n','A: ', doc['Answer']['Aliases'][0]])

    def evaluate(self, docs, lm):
        pass

