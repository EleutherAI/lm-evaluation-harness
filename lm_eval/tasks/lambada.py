from lm_eval.base import Task, rf, mean, perplexity
from lm_eval.utils import sh
import json
import math
from best_download import download_file


class LAMBADA(Task):
    def download(self):
        sh("mkdir -p data/lambada")
        download_file(
            "https://storage.googleapis.com/gpt-2/data/lambada_test.jsonl", 
            "data/lambada/lambada_test.jsonl", 
            "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"
        )

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
            'perplexity': ll,
            'accuracy': int(is_greedy)
        }
        
    def aggregation(self):
        return {
            'perplexity': perplexity,
            'accuracy': mean
        }

    def higher_is_better(self):
        return {
            'perplexity': False,
            'accuracy': True
        }
