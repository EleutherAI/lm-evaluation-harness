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
        instruction = f"""다음을 읽고 정답으로 알맞은 것을 고르시요.
### Context: {doc["context"]}
### Question: {doc["question"]}
### Options:
(1) {doc['option#1']}\n(2) {doc["option#2"]}\n(3) {doc["option#3"]}\n(4) {doc['option#4']}\n(5) {doc['option#5']}
### Answer: 주어진 문제의 정답은"""

        choices = [
            doc["option#1"],
            doc["option#2"],
            doc["option#3"],
            doc["option#4"],
            doc["option#5"],
        ]
        out_doc = {
            "question": instruction,
            "choices": ["(1)", "(2)", "(3)", "(4)", "(5)"],
            "gold": int(doc["gold"]) - 1,
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
