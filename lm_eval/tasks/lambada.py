import json
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
from lm_eval.utils import sh
from best_download import download_file
import os


class LAMBADA(Task):
    VERSION = 0
    def download(self):
        sh("mkdir -p data/lambada")
        try:
            if not os.path.exists("data/lambada/lambada_test.jsonl"):
                download_file(
                    "http://eaidata.bmk.sh/data/lambada_test.jsonl", 
                    "data/lambada/lambada_test.jsonl", 
                    "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"
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
    
    def fewshot_description(self):
        # TODO: figure out description
        return ""

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
