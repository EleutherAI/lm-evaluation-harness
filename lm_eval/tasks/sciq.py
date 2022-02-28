"""
Crowdsourcing Multiple Choice Science Questions
https://aclanthology.org/W17-4413.pdf

The SciQ dataset contains 13,679 crowdsourced science exam questions about Physics,
Chemistry and Biology, among others. The questions are in multiple-choice format
with 4 answer options each. For the majority of the questions, an additional paragraph
with supporting evidence for the correct answer is provided.

Homepage: https://allenai.org/data/sciq
"""
import os
import json
import zipfile
from lm_eval.base import MultipleChoiceTask
from best_download import download_file


_CITATION = """
@inproceedings{Welbl2017CrowdsourcingMC,
    title={Crowdsourcing Multiple Choice Science Questions},
    author={Johannes Welbl and Nelson F. Liu and Matt Gardner},
    booktitle={NUT@EMNLP},
    year={2017}
}
"""


class SciQ(MultipleChoiceTask):
    VERSION = 0
    # Multiple languages and multiple years
    def download(self):
        if not os.path.exists('data/sciq'):
            os.makedirs('data/sciq', exist_ok=True)
            download_file(
                'https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip',
                local_file='data/sciq/SciQ.zip',
                expected_checksum='7f3312f6ac6b09970b32942d106a8c44ec0dad46a0369f17d635aff8e348a87c',
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

    def training_docs(self):
        return self.load_docs("data/sciq/SciQ dataset-2 3/train.json")

    def validation_docs(self):
        return self.load_docs("data/sciq/SciQ dataset-2 3/valid.json")

    def test_docs(self):
        return self.load_docs("data/sciq/SciQ dataset-2 3/test.json")

    def doc_to_text(self, doc):
        return "{}\nQuestion: {}\nAnswer:".format(doc["source"], doc["query"]).strip()
