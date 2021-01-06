# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

import json
import random
import os
from lm_eval.base import Dataset
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

    # TODO: Implement evaluation code

    # ***IMPORTANT***: this evaluation function needs to be written for the new framework. 
    # For more info, check out the interface in base.py and the example BoolQ implementation in superglue.py. 
    # Remove this comment when the evaluation code is implemented.