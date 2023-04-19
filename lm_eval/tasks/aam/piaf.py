from .mlqa import MLQABase


class Piaf(MLQABase):
    VERSION = 0
    DATASET_PATH = "piaf"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["train"]  # using the train docs for validation

    def fewshot_description(self):
        return ""

    def doc_to_text(self, doc):
        return (
            "Contexte: "
            + doc["context"]
            + "\n\n"
            + "Question: "
            + doc["question"]
            + "\n\n"
            + "Réponse:"
        )
