from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
import re
from typing import List
import numpy as np


class BasedDrop(ConfigurableTask):
    VERSION = "default"
    DATASET_PATH = "hazyresearch/based_drop"
    DATASET_NAME = None

    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        context = doc["context"].strip()
        question = doc["question"].strip()
        while(context.lower().endswith(question.lower())):
            context = context[:-len(question)]

        out = (
            context.strip().strip(".") + ". " + question
        )
        return out

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc['answers']
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = "unanswerable"
        return " " + answer

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"], "max_new_tokens": 48}),
                idx=0,
                **kwargs,
            ),
        ]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """

        continuation = results[0]

        return {
            "contains": contains_score(continuation, doc["answers"])
        }


    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "contains": np.mean,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "contains": True
        }

def contains_score(prediction: str, labels: List[str]):
    return max(
        int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction)))
        for label in labels
    )

class BasedDropTwice(BasedDrop):

    def doc_to_text(self, doc):
        context = doc["context"].strip()
        question = doc["question"].strip()
        while(context.lower().endswith(question.lower())):
            context = context[:-len(question)]

        out = context.strip().strip(".").strip() + "."
        out += "\n" + out + " " + question

        intro_q = doc['orig_question'].strip(":")
        out = f"{intro_q} " + out
        return out