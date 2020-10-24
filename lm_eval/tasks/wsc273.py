import json
import random
import os
from lm_eval.base import Dataset
from ..utils import sh


class WinogradSchemaChallenge273(Dataset):    
    def __init__(self):
        super().__init__()

    def download(self):
        if not os.path.exists('data/wsc273'):
            sh("""
                mkdir -p data/wsc273 
                wget https://git.cse.msu.edu/bakerb15/nlp-final-project/raw/master/Winogard/reproduce/commonsense_test/wsc273.json -O data/wsc273/wsc273.json
                """)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return []

    def validation_docs(self):
        return []

    def test_docs(self):
        myjson = json.load(open('data/wsc273/wsc273.json'))
        return self.load_doc(myjson)
    
    def fewshot_description(self):
        # This format is ONLY for the purposes of deduplication. For the task evaluation, we'll need to find a new strategy,
        # to meet the needs of this particular task.
        return "Winograd schema sentence with correct continuation. True. Winograd schema sentence with incorrect continuation. False."

    def load_doc(self, myjson):
        docs = []
        for i in range(0, 273 * 2, 2):
            item1 = myjson[i]
            item2 = myjson[i+1]

            if item1['question_id'] != item2['question_id']:
                raise ValueError("WSC273 has missing completion pair.")

            question_id = item1['question_id']

            if item1['correctness'] == True:
                doc = {
                    'id': question_id,
                    'completions': {
                        'T': item1['substitution'],
                        'F': item2['substitution'],
                    },
                }
                
            if item2['correctness'] == True:
                doc = {
                    'id': question_id,
                    'completions': {
                        'F': item1['substitution'],
                        'T': item2['substitution'],
                    },
                }

            docs.append(doc)
 
        return docs
    
    def doc_to_text(self, doc, include_target=True):
        # WSC273 is currently only writing out full examples. Partial evaluation needs implementing.
        text = doc['completions']['T'] + ' True. ' + doc['completions']['F'] + ' False.'
        return text

    def evaluate(self, docs, lm):
        # TODO: Write evaluation function
        raise NotImplementedError()
