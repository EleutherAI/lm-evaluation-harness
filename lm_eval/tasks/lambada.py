# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

from lm_eval.base import Dataset
from lm_eval.utils import sh
import json
import requests
import ftfy


class Lambada(Dataset):
    def __init__(self):
        self.download()
    def download(self):
        sh("mkdir -p data/lambada")
        with open("data/lambada/lambada_test.json", 'w') as f:
            req = requests.get("https://storage.googleapis.com/gpt-2/data/lambada_test.jsonl")
            req.raise_for_status()
            jsons = [json.loads(l) for l in req.iter_lines()]
            texts = [ftfy.fix_text(j['text'], normalization='NFKC') for j in jsons]
            json.dump(texts, f)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def load_doc(self, myjson):
        return [doc for doc in myjson]

    def test_docs(self):
        myjson = json.load(open("data/lambada/lambada_test.json"))
        return self.load_doc(myjson)

    def doc_to_text(self, doc, include_target=True):
        #TODO: check if this is how OA does it
        #label = doc[]
        return doc

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        pass