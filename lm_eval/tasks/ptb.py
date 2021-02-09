from lm_eval.base import Task, rf, mean, perplexity
from lm_eval.utils import sh
import json
import math
from best_download import download_file

def detokenize(string):
    string = string.replace(" '", "'")
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" n't", "n't")
    string = string.replace(" N ", " 1 ")
    string = string.replace("$ 1", "$1")
    string = string.replace("# 1", "#1")
    return string


class PennTreebank(Task):
    def download(self):
        sh("mkdir -p data/ptb")
        download_file(
            "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt", 
            "data/ptb/ptb.train.txt",
            "fcea919f6cf83f35d4d00c6cbf08040d13d4155226340912e2fef9c9c4102cbf"
        )
        download_file(
            "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt", 
            "data/ptb/ptb.valid.txt",
            "c9fe6985fe0d4ccb578183407d7668fc6066c20700cb4cf87d8ff1cc34df1bf2"
        )
        download_file(
            "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt", 
            "data/ptb/ptb.test.txt",
            "dd65dff31e70846b2a6030a87482edcd5d199130cdcfa1f3dccbb033728deee0"
        )

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        with open("data/ptb/ptb.train.txt") as fh:
            for line in fh:
                yield line.strip()

    def validation_docs(self):
        with open("data/ptb/ptb.valid.txt") as fh:
            for line in fh:
                yield line.strip()

    def test_docs(self):
        with open("data/ptb/ptb.test.txt") as fh:
            for line in fh:
                yield line.strip()

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return detokenize(doc)

    
    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def construct_requests(self, doc, ctx):
        ll, _ = rf.loglikelihood(ctx, self.doc_to_target(doc))

        return ll,
    
    def process_results(self, doc, results):
        ll, = results
        
        return {
            'ppl': ll / len(doc.split(' '))
        }
        
    def aggregation(self):
        return {
            'ppl': perplexity
        }

    def higher_is_better(self):
        return {
            'ppl': False
        }
