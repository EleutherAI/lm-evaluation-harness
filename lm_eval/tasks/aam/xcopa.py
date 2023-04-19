import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


"""
    XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
    - https://ducdauge.github.io/files/xcopa.pdf
    - https://huggingface.co/datasets/xcopa
"""


class XCOPA_it(Task):
    VERSION = 0
    DATASET_PATH = "xcopa"
    DATASET_NAME = "it"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def fewshot_description(self):
        return "Data una premessa e un'alternativa con una relazione causale con la premessa e un'altra senza, scegliere l'alternativa più plausibile."

    def doc_to_text(self, doc):
        # modeled after the super_glue/copa evaluation
        # Drop the period
        connector = {
            "cause": "perché",
            "effect": "dunque",
        }[doc["question"]]
        return doc["premise"].strip()[:-1] + f" {connector}"

    def doc_to_target(self, doc):
        correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
        # Connect the sentences
        return " " + self.convert_choice(correct_choice)

    def construct_requests(self, doc, ctx):
        choice1 = " " + self.convert_choice(doc["choice1"])
        choice2 = " " + self.convert_choice(doc["choice2"])
        ll_choice1, _ = rf.loglikelihood(ctx, choice1)
        ll_choice2, _ = rf.loglikelihood(ctx, choice2)
        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0
        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    @staticmethod
    def convert_choice(choice):
        # remove upper case at the beginning of the sentence
        return choice[0].lower() + choice[1:]
