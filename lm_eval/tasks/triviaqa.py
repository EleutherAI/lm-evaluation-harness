import os
import json
from lm_eval.base import Task, rf
from ..metrics import mean
from ..utils import sh


class TriviaQA(Task):
    def download(self):
        if not os.path.exists('data/triviaqa'):
            sh("""
            mkdir -p data/triviaqa
            wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz -O data/triviaqa/trivia_qa-unfiltered.tar.gz
            tar -xf data/triviaqa/trivia_qa-unfiltered.tar.gz
            mv triviaqa-unfiltered/ data/triviaqa/
            """)
            
            # convert to streamable jsonl
            for subset in ['train', 'dev']:
                with open(f'data/triviaqa/triviaqa-unfiltered/unfiltered-web-{subset}.jsonl', 'w') as fh:
                    for d in json.load(open(f'data/triviaqa/triviaqa-unfiltered/unfiltered-web-{subset}.json'))['Data']:
                        fh.write(json.dumps(d) + "\n")

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return map(json.loads, open('data/triviaqa/triviaqa-unfiltered/unfiltered-web-train.jsonl'))

    def validation_docs(self):
        return map(json.loads, open('data/triviaqa/triviaqa-unfiltered/unfiltered-web-val.jsonl'))

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
