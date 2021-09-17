from . common import HFTask
from lm_eval.base import rf
from ..metrics import mean


class WebQs(HFTask):
    VERSION = 0
    DATASET_PATH = "web_questions"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        return "Question: " + doc['question'] + '\nAnswer:'

    def doc_to_target(self, doc):
        # this picks one answer to be the "correct" one, despite sometimes 
        # multiple correct answers being possible.
        # TODO: make sure we're actually handling multi-answer correctly
        return " " + doc['answers'][0]
        
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
        for alias in self._remove_prefixes(doc['answers']):
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