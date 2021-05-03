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


class PileArxivPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "ArXiv"


class PileBooks3PerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Books3"


class PileBookCorpus2PerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "BookCorpus2"


class PileCommonCrawlPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "CommonCrawl"


class PileDmMathematicsPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "DM Mathematics"


class PileEnronPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Enron Emails"


class PileEuroparlPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "EuroParl"


class PileFreeLawPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "FreeLaw"


class PileGithubPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Github"


class PileGutenbergPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Gutenberg (PG-19)"


class PileHackernewsPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "HackerNews"


class PileNIHExporterPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "NIH ExPorter"


class PileOpenSubtitlesPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "OpenSubtitles"


class PileOpenWebText2PerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "OpenWebText2"


class PilePhilPapersPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "PhilPapers"


class PilePileCcPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Pile-CC"


class PilePubmedAbstractsPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "PubMed Abstracts"


class PilePubmedCentralPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "PubMed Central"


class PileStackExchangePerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "StackExchange"


class PileUsptoPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "USPTO Backgrounds"


class PileUbuntuIrcPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Ubuntu IRC"


class PileWikipediaPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "Wikipedia (en)"


class PileYoutubeSubtitlesPerplexityTask(PilePerplexityTask):
    PILE_SET_NAME = "YoutubeSubtitles"
