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
        # TODO: separate pile val/test out by component so we don't have to scan the entire file once per set
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


class PileArxiv(PilePerplexityTask):
    PILE_SET_NAME = "ArXiv"


class PileBooks3(PilePerplexityTask):
    PILE_SET_NAME = "Books3"


class PileBookCorpus2(PilePerplexityTask):
    PILE_SET_NAME = "BookCorpus2"


class PileDmMathematics(PilePerplexityTask):
    PILE_SET_NAME = "DM Mathematics"


class PileEnron(PilePerplexityTask):
    PILE_SET_NAME = "Enron Emails"


class PileEuroparl(PilePerplexityTask):
    PILE_SET_NAME = "EuroParl"


class PileFreeLaw(PilePerplexityTask):
    PILE_SET_NAME = "FreeLaw"


class PileGithub(PilePerplexityTask):
    PILE_SET_NAME = "Github"


class PileGutenberg(PilePerplexityTask):
    PILE_SET_NAME = "Gutenberg (PG-19)"


class PileHackernews(PilePerplexityTask):
    PILE_SET_NAME = "HackerNews"


class PileNIHExporter(PilePerplexityTask):
    PILE_SET_NAME = "NIH ExPorter"


class PileOpenSubtitles(PilePerplexityTask):
    PILE_SET_NAME = "OpenSubtitles"


class PileOpenWebText2(PilePerplexityTask):
    PILE_SET_NAME = "OpenWebText2"


class PilePhilPapers(PilePerplexityTask):
    PILE_SET_NAME = "PhilPapers"


class PilePileCc(PilePerplexityTask):
    PILE_SET_NAME = "Pile-CC"


class PilePubmedAbstracts(PilePerplexityTask):
    PILE_SET_NAME = "PubMed Abstracts"


class PilePubmedCentral(PilePerplexityTask):
    PILE_SET_NAME = "PubMed Central"


class PileStackExchange(PilePerplexityTask):
    PILE_SET_NAME = "StackExchange"


class PileUspto(PilePerplexityTask):
    PILE_SET_NAME = "USPTO Backgrounds"


class PileUbuntuIrc(PilePerplexityTask):
    PILE_SET_NAME = "Ubuntu IRC"


class PileWikipedia(PilePerplexityTask):
    PILE_SET_NAME = "Wikipedia (en)"


class PileYoutubeSubtitles(PilePerplexityTask):
    PILE_SET_NAME = "YoutubeSubtitles"
