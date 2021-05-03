import os

import lm_dataformat
import abc
import numpy as np
from lm_eval.base import rf, PerplexityTask
from ..metrics import mean, matthews_corrcoef, f1_score
from ..utils import general_detokenize
from best_download import download_file


class PilePerplexityTask(PerplexityTask, abc.ABC):

    PILE_SET_NAME = None
    VAL_PATH = 'data/pile/val.jsonl.zst'
    TEST_PATH = 'data/pile/test.jsonl.zst'

    def download(self):
        os.makedirs("data/pile/", exist_ok=True)
        if not os.path.exists(self.VAL_PATH):
            download_file("https://the-eye.eu/public/AI/pile/val.jsonl.zst", self.VAL_PATH)
        if not os.path.exists(self.TEST_PATH):
            download_file("https://the-eye.eu/public/AI/pile/test.jsonl.zst", self.TEST_PATH)

    def validation_docs(self):
        rdr = lm_dataformat.Reader(self.VAL_PATH)
        for doc, metadata in rdr.stream_data(get_meta=True):
            if metadata["pile_set_name"] == self.PILE_SET_NAME:
                yield doc

    def test_docs(self):
        rdr = lm_dataformat.Reader(self.TEST_PATH)
        for doc, metadata in rdr.stream_data(get_meta=True):
            if metadata["pile_set_name"] == self.PILE_SET_NAME:
                yield doc

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True


class PileEnronPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Enron Emails"


class PileUbuntuPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Ubuntu IRC"
