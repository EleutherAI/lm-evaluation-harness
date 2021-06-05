import os
import json
import zipfile
from lm_eval.base import MultipleChoiceTask
from best_download import download_file


class SciQ(MultipleChoiceTask):
    VERSION = 0
    # Multiple languages and multiple years
    def download(self):
        if not os.path.exists('data/sciq'):
            os.mkdir('data/sciq')
            download_file(
                'https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip',
                'data/sciq/SciQ.zip', 
                '7f3312f6ac6b09970b32942d106a8c44ec0dad46a0369f17d635aff8e348a87c',
            )
            with zipfile.ZipFile("data/sciq/SciQ.zip", "r") as zf:
                zf.extractall("data/sciq/")

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        choices = [
            doc["distractor1"], 
            doc["distractor2"], 
            doc["distractor3"],
            doc["correct_answer"],
        ]
        src = doc['support']
        out_doc = {
            "source" : src,
            "query" : doc['question'],
            "choices" : choices,
            "gold" : 3,
        }
        return out_doc
    
    def load_docs(self, textfilename):
        with open(textfilename, 'r') as j:
            docs = json.loads(j.read()) 
        for record in docs:
            yield self._convert_standard(record)

    def fewshot_description(self):
        return ""

    def training_docs(self):
        return self.load_docs("data/sciq/SciQ dataset-2 3/train.json")

    def validation_docs(self):
        return self.load_docs("data/sciq/SciQ dataset-2 3/valid.json")

    def test_docs(self):
        return self.load_docs("data/sciq/SciQ dataset-2 3/test.json")

    def doc_to_text(self, doc):
        return "{}\nQuestion: {}\nAnswer:".format(doc["source"], doc["query"]).strip()