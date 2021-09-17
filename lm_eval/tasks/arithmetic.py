import abc
import json
import os
from collections import namedtuple
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from best_download import download_file

ArithmeticDoc = namedtuple('ArithmeticDoc', ['context', 'completion'])


class Arithmetic(Task):
    VERSION = 0
    directory = 'data/arithmetic/'

    def __init__(self):
        super().__init__()

    def download(self):
        file_name, checksum = self.get_file_download_info()
        url = 'https://raw.githubusercontent.com/openai/gpt-3/master/data/' + file_name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        download_file(url, self.directory+file_name, checksum)
        self.set_docs()

    @abc.abstractmethod
    def get_file_download_info(self):
        """returns a tuple of (file_name, checksum)"""
        pass

    def set_docs(self):
        file_name, _ = self.get_file_download_info()
        jsons = open(self.directory+file_name, 'r')
        self._docs = [self.load_doc(json.loads(line)) for line in jsons]

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return NotImplemented

    def validation_docs(self):
        return self._docs

    def test_docs(self):
        return NotImplemented
    
    def doc_to_text(self, doc):
        return doc.context

    def doc_to_target(self, doc):
        return doc.completion

    def load_doc(self, doc_json):
        return ArithmeticDoc(context=doc_json['context'].strip()
            .replace('\n\n', '\n')
            .replace('Q:', 'Question:')
            .replace('A:', 'Answer:'), completion=doc_json['completion'])
    
    def construct_requests(self, doc, ctx):
        ll, is_prediction = rf.loglikelihood(ctx, doc.completion)
        return is_prediction

    def process_results(self, doc, results):
        is_prediction, = results
        return {
            "acc": is_prediction
        }

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {
            "acc": True
        }


class Arithmetic2DPlus(Arithmetic):
    def get_file_download_info(self):
        return 'two_digit_addition.jsonl', '75a54b7a3db3b23369df74fe440c23025f3d3c51f664300bd3d56632b2617b3d'

class Arithmetic2DMinus(Arithmetic):
    def get_file_download_info(self):
        return 'two_digit_subtraction.jsonl', 'da956066ff108c00b341d360567472784f5fd872d6465071b44a14291205bc03'

class Arithmetic3DPlus(Arithmetic):
    def get_file_download_info(self):
        return 'three_digit_addition.jsonl', '124865e30efd2abfbc1855dd34c218fc02d32d780ace970ab9b4ea3fa74c798b'

class Arithmetic3DMinus(Arithmetic):
    def get_file_download_info(self):
        return 'three_digit_subtraction.jsonl', '7fc6aaedcb0e2bd17c398dd4147c5585b1e608278a8e98b914e69656707d6a29'

class Arithmetic4DPlus(Arithmetic):
    def get_file_download_info(self):
        return 'four_digit_addition.jsonl', '459c6f75baa2e8d7cf50bdd07db6d0ca9133a6b137d95d09267db85b6e07f391'

class Arithmetic4DMinus(Arithmetic):
    def get_file_download_info(self):
        return 'four_digit_subtraction.jsonl', '0c47db40a10c052ef0cf732a9ef2edaa53d66377d43eb47a9c382d33a8af7102'

class Arithmetic5DPlus(Arithmetic):
    def get_file_download_info(self):
        return 'five_digit_addition.jsonl', '30ada42efe315b958c6e9649274005d3b720e50298e92c3a2d321f8996e58f54'

class Arithmetic5DMinus(Arithmetic):
    def get_file_download_info(self):
        return 'five_digit_subtraction.jsonl', '8b98ccfc943cbf9193bcf1984954aa0b1a4527016072d972a2b055cc1482ca3c'

class Arithmetic2DMultiplication(Arithmetic):
    def get_file_download_info(self):
        return 'two_digit_multiplication.jsonl', '5613d1d1cc3b2c03edc1990252247d34c10ec82944b2cdeb19e71b00f237f431'

class Arithmetic1DComposite(Arithmetic):
   def get_file_download_info(self):
        return 'single_digit_three_ops.jsonl', '08b34e3272a8ff1d4932d63f251519d14c485c38d582366e1e323d0b859c3925'
