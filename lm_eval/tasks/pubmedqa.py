"""
"""

import os
import numpy as np
import json
from ..utils import sh
from .common import HFTask, yesno
from lm_eval.base import MultipleChoiceTask, rf, mean
import zipfile


class Pubmed_QA(HFTask):
    DATASET_PATH = "pubmed_qa"
    DATASET_NAME = "pqa_labeled"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # Average ctx length in labelled dataset is 238.9
        # 2 few-shot exmamples pushes it beyond context window
        return ""

    def doc_to_text(self, doc):
        ctxs = "\n".join(doc["context"]["contexts"])
        return "abstract: {}\nquestion: {}\nanswer:".format(
            ctxs,
            doc["question"],
            doc["final_decision"]
        )

    def doc_to_target(self, doc):
        return " {}".format(doc["final_decision"])

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns
        an iterable of Requests which will be sent to the LM.
        """
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        ll_maybe, _ = rf.loglikelihood(ctx, " maybe")
        return ll_yes, ll_no, ll_maybe

    def process_results(self, doc, results):
        gold = doc["final_decision"]
        ll_yes, ll_no, ll_maybe = results
        pred = np.argmax(results)
        return {
            "acc": ["yes", "no", "maybe"][pred] == gold, 
        }

    def aggregation(self):
        return {
            "acc" : mean
        }

    def higher_is_better(self):
        return {
            "acc" : True
        }


class SciQ(MultipleChoiceTask):
    def download(self):
        if not os.path.exists('data/sciq'):
            os.mkdir('data/sciq')
            sh((
                "wget https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip -O data/sciq/SciQ.zip"
            ))
            with zipfile.ZipFile("data/sciq/SciQ.zip", "r") as zf:
                zf.extractall("data/sciq/")

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_strandard(doc):
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
        if labelfilename != None:
            with open(textfilename, 'r') as j:
                docs = json.loads(j.read()) 
        for record in docs:
            yield _convert_standard(record)

    def fewshot_description(self):
        # Average ctx length in labelled dataset is 238.9
        # 2 few-shot exmamples pushes it beyond context window
        return ""

    def training_docs(self):
        return self.load_docs("data/sciq/Sci-Q\ dataset-2\ 3/train.json")

    def validation_docs(self):
        return self.load_docs("data/sciq/Sci-Q\ dataset-2\ 3/valid.json")

    def test_docs(self):
        return self.load_docs("data/sciq/Sci-Q\ dataset-2\ 3/test.json")

    def doc_to_text(self, doc):
        return " {}\n{}".format(doc["source"], doc["query"])

class EmrQA():
    def load_docs(self, textfilename):
        pass
