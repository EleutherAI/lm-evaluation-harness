# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

import json
import random
import os
from lm_eval.base import Dataset, rf, mean
from tqdm import auto as tqdm_lib
from . common import simple_accuracy_metric
import numpy as np
from ..utils import sh


class SATAnalogies(Dataset):    
    def __init__(self):
        super().__init__()

    def download(self):
        # We should be using a checksum here.
        # The canonical sha256 hash is below:
        # 9dece377d8d57253ef8c78370ff15de0bb1d9e90a82c815a67ba1e621e921bfc
        if not os.path.exists('data/sat') and os.path.exists('data/sat/SAT-package-V3.txt'):
            raise NotImplementedError('SAT Analogies dataset is not provided. Follow instructions on https://aclweb.org/aclwiki/SAT_Analogy_Questions_(State_of_the_art) to locate.')

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
        data = []

        with open("data/sat/SAT-package-V3.txt", "r") as f:
            lines = f.read().splitlines() 
            record = []
            for line in lines:
                if len(line) == 0 and record:
                    data.append(record)
                    record = []
                elif len(line) > 0 and line[0] == '#':
                    continue
                else:
                    record.append(line)
            data.append(record)

        docs = []

        for record in data:
            source = record[-8]
            query = record[-7]
            choices = record[-6:-1]
            answer_key = record[-1]

            doc = {
                'source': source,
                'query': query,
                'choices': choices,
                'answer_key': answer_key,
            }
            docs.append(doc)

        return docs

    
    def fewshot_description(self):
        # This format is ONLY for the purposes of deduplication. For the task evaluation, we'll need to find a new strategy,
        # to meet the needs of this particular task.
        return "first thing is to second thing as\nthird thing is to fourth thing\nfifth thing is to sixth thing\nseventh thing is to eighth thing\nninth thing is to tenth thing\neleventh thing is to twelfth thing\nanswer which is either a b c d or e"

    def doc_to_text(self, doc, include_target=True):
        # SAT Analogies is currently only writing out full examples. Partial evaluation needs implementing.
        format_qn = lambda x: x[0] + ' is to ' + x[1]

        query = doc['query']
        choices = doc['choices']
        answer = doc['answer_key']

        query_words = query.split(' ')[:2]
        text = format_qn(query_words) + ' as' + '\n'

        for choice in choices:
            choice_words = choice.split(' ')[:2]
            text += format_qn(choice_words) + '\n'

        if include_target:
            text += answer

        return text

    def doc_to_target(self, doc):
        # assumes answer_key is the true-answer's letter
        return doc['answer_key']

    def construct_requests(self, ctx):
        # assumes the output is the predicted-answer's letter
        ll_a = rf.loglikelihood(ctx, ' a')
        ll_b = rf.loglikelihood(ctx, ' b')
        ll_c = rf.loglikelihood(ctx, ' c')
        ll_d = rf.loglikelihood(ctx, ' d')
        ll_e = rf.loglikelihood(ctx, ' e')

        return ll_a, ll_b, ll_c, ll_d, ll_e

    def process_results(self, doc, results):
        predicted_odds = np.array(list(results))
        gold = doc["answer_key"]

        acc = 1. if np.argmax(predicted_odds) == gold else 0.

        return [
            {
                "submetric": "acc",
                "value": acc,
                "higher_is_better": True,
                "aggregation": mean
            }
        ]


    def evaluate(self, docs, lm):
        # functionality already implemented above
        raise NotImplementedError()
