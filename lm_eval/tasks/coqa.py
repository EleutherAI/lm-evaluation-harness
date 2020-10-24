import json
import random
from lm_eval.base import Dataset
from ..utils import sh


class CoQA(Dataset):
    def __init__(self):
        self.download()
    def download(self):
        #TODO: don't download if files already there
        sh("""
            mkdir -p data/coqa 
            wget http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json -O data/coqa/coqa-train-v1.0.json
            wget http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json -O data/coqa/coqa-dev-v1.0.json
            """)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return json.load(open('data/coqa/coqa-train-v1.0.json'))['data']

    def validation_docs(self):
        return  json.load(open('data/coqa/coqa-dev-v1.0.json'))['data']  

    def test_docs(self):
        pass   
    
    def fewshot_description(self):
        pass
    
    def doc_to_text(self, doc, include_target=True):
        text = [doc['story']]
        for pair in zip(doc['questions'], doc['answers']):
            text.append('\n\n')
            text.append(''.join(['Q: ',pair[0]['input_text'], '\n\n']))
            text.append(''.join(['A: ',pair[1]['input_text']]))

        return ''.join(text)

    def evaluate(self, docs, lm):
        pass
