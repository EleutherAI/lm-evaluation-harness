"""
ASDiv: A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers
https://arxiv.org/abs/2106.15772

@misc{miao2021diverse,
      title={A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers},
      author={Shen-Yun Miao and Chao-Chun Liang and Keh-Yih Su},
      year={2021},
      eprint={2106.15772},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
"""
from lm_eval.base import Task
from pathlib import Path
from best_download import download_file 
import xml.etree.ElementTree as ET
from lm_eval.base import rf
from lm_eval.metrics import mean
import numpy as np
from zipfile import ZipFile
import os 

#currently ignoring formula for answer generation

# given a subset, splits return the docs 
class Asdiv(Task):
    VERSION = 0
    DATASET_PATH = Path("data/asdiv")

    def download(self):
        if self.DATASET_PATH.exists():
            return
        Path.mkdir(self.DATASET_PATH)
        url = "https://github.com/chaochun/nlu-asdiv-dataset/archive/refs/heads/master.zip"
        checksum = "2f71f8003929d605369ad924be4b95c15879fc2bfac0d4d01a81f8aabceaad5c"
        zip_path = self.DATASET_PATH / "master.zip"
        download_file(url, str(zip_path), checksum)
        with ZipFile(zip_path, "r") as zip:
            zip.extractall(self.DATASET_PATH)
        os.remove(zip_path)

    def _convert_standard(self, problem):
        #TODO: include solution-type and formula
        out_doc = {
            "question" : problem.find('Question').text,
            "body" : problem.find('Body').text,
            "answer": problem.find('Answer').text
        }
        return out_doc

    def load_docs(self, textfilename, tfds=False):
        tree = ET.parse(textfilename)
        root = tree.getroot()
        for pid, problem in enumerate(root.iter('Problem')):
            out_doc = self._convert_standard(problem)
            yield out_doc

    def has_training_docs(self):
        return False
    
    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        raise NotImplementedError("This dataset has no training docs")

    def test_docs(self):
        raise NotImplementedError("This dataset has no test docs")

    def validation_docs(self):
        data_xml_path = self.DATASET_PATH / "nlu-asdiv-dataset-master/dataset/ASDiv.xml"
        return self.load_docs(data_xml_path)

    def fewshot_context(self, doc, num_fewshot, provide_description, rnd):
        assert num_fewshot == 0, "ASDiv is intended only for the zero-shot setting."
        return super().fewshot_context(doc, num_fewshot, provide_description, rnd)

    
    def fewshot_description(self):
        # TODO: add solution-type and formula
        desc = "information containing the context of the question\nQuestion: Text of a question.\nAnswer: Answer to the question, based on the passage.\n"
        return desc

    def doc_to_text(self, doc):
        # TODO: add solution-type
        return doc['body'] + '\n' + 'Question:' + doc['question'] + '\n' + 'Answer:'

    def doc_to_target(self, doc):
        # TODO: add formula
        return doc['answer']

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.
        :param doc:
        The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
        The context string, generated by fewshot_context. This includes the natural
        language description, as well as the few shot examples, and the question
        part of the document for `doc`.
        """
        return rf.greedy_until(ctx, ["\n"])

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document
        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        retval = 0
        if self.is_equiv(doc['answer'], results[0]):
            retval = 1
        return {
            "acc": retval
        }

    def is_equiv(self, str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False
        return str1 == str2


    def aggregation(self):
        return {
            'acc': mean
        }

    def higher_is_better(self):
        return {
            'acc': True
        }








    

