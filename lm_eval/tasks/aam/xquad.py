from .mlqa import MLQABase


class XQUADBase(MLQABase):
    VERSION = 0
    DATASET_PATH = "xquad"

    def fewshot_description(self):
        return ""

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False


class XQUADDe(XQUADBase):
    DATASET_NAME = "xquad.de"

    def doc_to_text(self, doc):
        return (
            "Kontext: "
            + doc["context"]
            + "\n\n"
            + "Frage: "
            + doc["question"]
            + "\n\n"
            + "Antwort:"
        )


class XQUADEn(XQUADBase):
    DATASET_NAME = "xquad.en"

    def doc_to_text(self, doc):
        return (
            "Context: "
            + doc["context"]
            + "\n\n"
            + "Question: "
            + doc["question"]
            + "\n\n"
            + "Answer:"
        )


class XQUADEs(XQUADBase):
    DATASET_NAME = "xquad.es"

    def doc_to_text(self, doc):
        return (
            "Contexto: "
            + doc["context"]
            + "\n\n"
            + "Pregunta: "
            + doc["question"]
            + "\n\n"
            + "Respuesta:"
        )
