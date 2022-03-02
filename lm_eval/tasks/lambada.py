"""
The LAMBADA dataset: Word prediction requiring a broad discourse context∗
https://arxiv.org/pdf/1606.06031.pdf

LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI
"""
import json
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
from lm_eval.utils import sh
from best_download import download_file
import os


_CITATION = """
@misc{
    author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and Baroni, Marco and Boleda, Gemma and Fernández, Raquel}, 
    title={The LAMBADA dataset},
    DOI={10.5281/zenodo.2630551},
    publisher={Zenodo},
    year={2016},
    month={Aug}
}
"""


class LAMBADA(Task):
    VERSION = 0
    def download(self):
        sh("mkdir -p data/lambada")
        try:
            if not os.path.exists("data/lambada/lambada_test.jsonl"):
                download_file(
                    "http://eaidata.bmk.sh/data/lambada_test.jsonl", 
                    local_file="data/lambada/lambada_test.jsonl",
                    expected_checksum="4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"
                )
        except:
            # fallback - for some reason best_download doesnt work all the time here
            sh("wget http://eaidata.bmk.sh/data/lambada_test.jsonl -O data/lambada/lambada_test.jsonl")
            sh('echo "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226  data/lambada/lambada_test.jsonl" | sha256sum --check')

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        pass

    def validation_docs(self):
        with open("data/lambada/lambada_test.jsonl") as fh:
            for line in fh:
                yield json.loads(line)

    def test_docs(self):
        pass

    def doc_to_text(self, doc):
        return doc['text'].rsplit(' ', 1)[0]

    def doc_to_target(self, doc):
        return " " + doc['text'].rsplit(' ', 1)[1]

    def construct_requests(self, doc, ctx):
        ll, is_greedy = rf.loglikelihood(ctx, self.doc_to_target(doc))

        return ll, is_greedy
    
    def process_results(self, doc, results):
        ll, is_greedy = results

        return {
            'ppl': ll,
            'acc': int(is_greedy)
        }
        
    def aggregation(self):
        return {
            'ppl': perplexity,
            'acc': mean
        }

    def higher_is_better(self):
        return {
            'ppl': False,
            'acc': True
        }
