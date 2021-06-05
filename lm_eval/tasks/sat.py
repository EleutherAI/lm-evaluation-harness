import os
from lm_eval.base import MultipleChoiceTask


class SATAnalogies(MultipleChoiceTask):    
    VERSION = 0
    NEEDS_MANUAL_DL = True
    
    def __init__(self):
        super().__init__()

    def download(self):
        # We should be using a checksum here.
        # The canonical sha256 hash is below:
        # 9dece377d8d57253ef8c78370ff15de0bb1d9e90a82c815a67ba1e621e921bfc

        if not os.path.exists('data/sat/SAT-package-V3.txt'):
            raise NotImplementedError('SAT Analogies dataset is not provided. Follow instructions on https://aclweb.org/aclwiki/SAT_Analogy_Questions_(State_of_the_art) to locate.')

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return []
    def test_docs(self):
        return []

    def validation_docs(self):
        data = []

        with open("data/sat/SAT-package-V3.txt", "r") as f:
            record = []
            for line in f:
                line = line.strip()
                if len(line) == 0 and record:
                    data.append(record)
                    record = []
                elif len(line) > 0 and line[0] == '#':
                    continue
                else:
                    record.append(line)
            data.append(record)

        for record in data:
            source = record[-8]
            query = record[-7]
            choices = record[-6:-1]
            answer_key = record[-1]

            doc = {
                'source': source,
                'query': query.split(' ')[:2],
                'choices': ["{} is to {}".format(*c.split(' ')[:2]) for c in choices],
                'gold': ['a','b','c','d','e'].index(answer_key.strip()),
            }
            yield doc

    
    def fewshot_description(self):
        # TODO: figure out actual description
        return ""

    def doc_to_text(self, doc):
        return "{} is to {} as".format(*doc['query'])
