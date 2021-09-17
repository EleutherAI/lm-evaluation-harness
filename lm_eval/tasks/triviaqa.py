import os
import json
import jsonlines
from lm_eval.base import Task, rf
from ..metrics import mean
from ..utils import sh
from best_download import download_file


class TriviaQA(Task):
    VERSION = 0
    def download(self):
        if not os.path.exists('data/triviaqa/unfiltered-web-train.jsonl'):
            os.makedirs("data/triviaqa/", exist_ok=True)
            download_file("http://eaidata.bmk.sh/data/triviaqa-unfiltered.tar.gz", "data/triviaqa/triviaqa-unfiltered.tar.gz", "adc19b42769062d241a8fbe834c56e58598d9322eb6c614e9f33a68a2cf5523e")
            sh("""
            cd data/triviaqa/
            tar -xf triviaqa-unfiltered.tar.gz
            """)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return jsonlines.open('data/triviaqa/unfiltered-web-train.jsonl')

    def validation_docs(self):
        return jsonlines.open('data/triviaqa/unfiltered-web-dev.jsonl')

    def test_docs(self):
        raise NotImplementedError()
    
    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""
    
    def doc_to_text(self, doc):
        return f"Question: {doc['Question']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc['Answer']['Value']

    def _remove_prefixes(self, aliases):
        # Optimization: Remove any alias that has a strict prefix elsewhere in the list
        # we can do this because if the prefix is acceptable by isgreedy, we can stop looking
        aliases.sort()
        ret = [aliases[0]]
        for alias in aliases[1:]:
            if not alias.startswith(ret[-1]):
                ret.append(alias)

        return ret
        

    def construct_requests(self, doc, ctx):
        ret = []
        for alias in self._remove_prefixes(doc['Answer']['Aliases']):
            _, is_prediction = rf.loglikelihood(ctx, " " + alias)
            ret.append(is_prediction)
        return ret

    def process_results(self, doc, results):
        return {
            "acc": float(any(results))
        }

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {
            "acc": True
        }
