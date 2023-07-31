from lm_eval.base import MultipleChoiceTask


class Haerae(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "amphora/haerae_bench"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])
    
    def _process_doc(self, doc):
        choices = [doc["o1"], doc["o2"], doc["o3"], doc["o4"]]
        if doc.get("o5") is not None:
            choices.append(doc["o5"])
        out_doc = {
            "query": doc["query"],
            "choices": choices,
            "gold": int(doc['gold'])-1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]


class HI(Haerae):
    DATASET_NAME = "HI"


class KGK(Haerae):
    DATASET_NAME = "KGK"


class LW(Haerae):
    DATASET_NAME = "LW"


class RC(Haerae):
    DATASET_NAME = "RC"


class RW(Haerae):
    DATASET_NAME = "RW"


class SN(Haerae):
    DATASET_NAME = "SN"
