from . common import HFTask
from lm_eval.base import MultipleChoiceTask


class HeadQABase(HFTask, MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "head_qa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        out_doc = {
            "id": doc["qid"],
            "query": "Question: " + doc["qtext"] + "\nAnswer:",
            "choices": [answer["atext"] for answer in doc["answers"]],
            "gold": int(doc["ra"]) - 1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

class HeadQAEn(HeadQABase):
    DATASET_NAME = "en"

class HeadQAEs(HeadQABase):
    DATASET_NAME = "es"

# for backwards compatibility
class HeadQAEsDeprecated(HeadQABase):
    DATASET_NAME = "es"

    def __init__(self):
        super().__init__()
        print("WARNING: headqa is deprecated. Please use headqa_es or headqa_en instead. See https://github.com/EleutherAI/lm-evaluation-harness/pull/240 for more info.")