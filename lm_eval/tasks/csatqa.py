from lm_eval.base import MultipleChoiceTask


class CSATQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "EleutherAI/csatqa"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])
    
    def _process_doc(self, doc):
        choices = [doc["option#1"], doc["option#2"], doc["option#3"], doc["option#4"], doc["option#5"]]
        out_doc = {
            "question": doc["question"],
            "choices": choices,
            "gold": int(doc['gold']),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["question"]


class WR(CSATQA):
    DATASET_NAME = "WR"
    
class GR(CSATQA):
    DATASET_NAME = "GR"

class RCS(CSATQA):
    DATASET_NAME = "RCS"
    
class RCSS(CSATQA):
    DATASET_NAME = "RCSS"
    
class RCH(CSATQA):
    DATASET_NAME = "RCH"

class LI(CSATQA):
    DATASET_NAME = "LI"
