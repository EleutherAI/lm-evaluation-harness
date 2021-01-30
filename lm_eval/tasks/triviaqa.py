import os
import json
import random
from lm_eval.base import Dataset, mean, rf
from ..utils import sh

class TriviaQA(Dataset):
    def download(self):
        if not os.path.exists('data/triviaqa'):
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
        # TODO: figure out fewshot description
        return ""
    
    def doc_to_text(self, doc):
        return ''.join(['Q:', doc['Question'], '\n\n','A:'])

    def doc_to_target(self, doc):
        return doc['Answer']['Aliases'][0]

    def construct_requests(self, doc, ctx):
        ll, is_prediction = rf.loglikelihood(ctx,doc['Answer']['Value'])
        return is_prediction

    def process_results(self, doc, results):
        is_prediction = results
        return {
            "acc": float(is_prediction[1])
        }

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {
            "acc": True
        }