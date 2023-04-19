from ..squad import SQuAD2


class SQUAD_IT(SQuAD2):
    VERSION = 1
    DATASET_PATH = "squad_it"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return (
            "Contesto: "
            + doc["context"]
            + "\n\n"
            + "Domanda: "
            + doc["question"]
            + "\n\n"
            + "Risposta:"
        )
