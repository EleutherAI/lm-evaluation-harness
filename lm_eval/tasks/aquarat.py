import numpy as np
from tqdm import auto as tqdm_lib
from . common import HFTask, simple_accuracy_metric, yesno
from lm_eval.base import rf, mean

class AquaRat():
    
    def __init__(self):
        self.download()
        
    def download(self):
        if os.path.exists('data/aquarat/'):
            pass
        else:
            sh( """
                mkdir -p data/aquarat
                wget https://raw.githubusercontent.com/deepmind/AQuA/master/train.json -O data/aquarat/aquarat-train.json
                wget https://raw.githubusercontent.com/deepmind/AQuA/master/dev.json -O data/aquarat/aquarat-valid.json
                wget https://raw.githubusercontent.com/deepmind/AQuA/master/test.json -O data/aquarat/aquarat-test.json
                """)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True
    
    def training_docs(self):
        doc_data = json.load(open('data/aquarat/aquarat-train.json')
        return doc_data

    def validation_docs(self):
        doc_data = json.load(open('data/aquarat/aquarat-valid.json')
        return doc_data

    def test_docs(self):
        doc_data = json.load(open('data/aquarat/aquarat-test.json')
        return doc_data

    def fewshot_description(self):
        # TODO: Incorporate rationale and compute using BLEU score
        return "Read the following multiple choice questions and pick the correct choice"

    def doc_to_text(self, doc):
        options = " ".join(doc['options'])
        return f"{doc['question']}\n{options}\nanswer: "
    
    def doc_to_target(self, doc):
        return doc['correct']
        
    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, doc['options'][i])
            for i in range(5)
        ]

        return lls

    def process_results(self, doc, results):
        gold = doc["correct"]

        acc = 1. if np.argmax(results) == gold else 0.

        return {
            "acc": acc
        }

    
    def higher_is_better(self):
        return {
            "acc": True
        }
    
    def aggregation(self):
        return {
            "acc": mean
        }
