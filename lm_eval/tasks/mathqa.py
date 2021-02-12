from . common import HFTask
from lm_eval.base import mean, rf

class MathQA(HFTask):
    DATASET_PATH = "math_qa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        return "Q: " + doc['Problem'] + '\nA:'

    def doc_to_target(self, doc):
        # this picks one answer to be the "correct" one, despite sometimes 
        # multiple correct answers being possible.
        # TODO: make sure we're actually handling multi-answer correctly
        return " " + doc['correct']
        
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

        self.answer_options = ['a', 'b', 'c', 'd', 'e']
        
        ret = []
        for i in range(len(self.answer_options)):
            ll, _ =rf.loglikelihood(ctx, ' ' + self.answer_options[i])
            ret.append(ll)

        return ret

    def process_results(self, doc, results):
        max_result_idx = max(enumerate(results), key=lambda x: x[1])[0]

        if doc['correct'] == self.answer_options[max_result_idx]:
            result = 1.0
        else:
            result = 0.0

        return {
            "acc": result
        }

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {
            "acc": True
        }