import numpy as np
from lm_eval.base import rf, mean
from . common import HFTask


class ARCEasy(HFTask):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Easy"

    letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def __init__(self):
        super().__init__()
        self.data = self.__clean_data()

    def __clean_data(self):
        """ Resolves various edge cases in the unprocessed HF ARC dataset. """
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}
        result = {}
        for split, data in self.data.items():
            result[split] = []
            for doc in data:
                # Ensure all `answerKey`s and `label`s are in letter format.
                doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
                doc["choices"]["label"] = [
                    num_to_letter.get(label, label) for label in doc["choices"]["label"]
                ]
                result[split].append(doc)
        return result

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
        return "Question: " + doc['question'] + '\nAnswer:'

    def doc_to_target(self, doc):
        index = self.letter_to_num[doc["answerKey"]]
        return " " + doc['choices']['text'][index]

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
        ll_choices = []
        for choice in doc["choices"]["text"]:
            ll_choices.append(rf.loglikelihood(ctx, " " + choice)[0])
        return ll_choices

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = self.letter_to_num[doc["answerKey"]]
        pred = np.argmax(results)
        return {
            "acc": pred == gold
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return {
            "acc": True
        }


class ARCChallenge(ARCEasy):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"
