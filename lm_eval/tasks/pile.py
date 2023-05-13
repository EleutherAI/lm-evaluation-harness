"""
The Pile: An 800GB Dataset of Diverse Text for Language Modeling
https://arxiv.org/pdf/2101.00027.pdf

The Pile is a 825 GiB diverse, open source language modelling data set that consists
of 22 smaller, high-quality datasets combined together. To score well on Pile
BPB (bits per byte), a model must be able to understand many disparate domains
including books, github repositories, webpages, chat logs, and medical, physics,
math, computer science, and philosophy papers.
Homepage: https://pile.eleuther.ai/
"""

from lm_eval.api.task import PerplexityTask

from lm_eval.api.register import register_task, register_group

_CITATION = """
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
"""


class PilePerplexityTask(PerplexityTask):
    VERSION = "2.0"
    DATASET_PATH = "EleutherAI/the_pile"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def test_docs(self):
        for doc in self.dataset["train"].select(range(100)):
            yield doc
    
    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_target(self, doc):
        return doc["text"]

    # def validation_docs(self):
    #     for doc in self.dataset["validation"]:
    #         yield doc["text"]

    # def test_docs(self):
    #     for doc in self.dataset["test"]:
    #         yield doc["text"]


class PileArxiv(PilePerplexityTask):
    DATASET_NAME = "pile_arxiv"


class PileBooks3(PilePerplexityTask):
    DATASET_NAME = "pile_books3"


class PileBookCorpus2(PilePerplexityTask):
    DATASET_NAME = "pile_bookcorpus2"


class PileDmMathematics(PilePerplexityTask):
    DATASET_NAME = "pile_dm-mathematics"


@register_task("pile_enron")
class PileEnron(PilePerplexityTask):
    DATASET_NAME = "enron_emails"


class PileEuroparl(PilePerplexityTask):
    DATASET_NAME = "pile_europarl"


class PileFreeLaw(PilePerplexityTask):
    DATASET_NAME = "pile_freelaw"


class PileGithub(PilePerplexityTask):
    DATASET_NAME = "pile_github"


class PileGutenberg(PilePerplexityTask):
    DATASET_NAME = "pile_gutenberg"


class PileHackernews(PilePerplexityTask):
    DATASET_NAME = "pile_hackernews"


class PileNIHExporter(PilePerplexityTask):
    DATASET_NAME = "pile_nih-exporter"


class PileOpenSubtitles(PilePerplexityTask):
    DATASET_NAME = "pile_opensubtitles"


class PileOpenWebText2(PilePerplexityTask):
    DATASET_NAME = "pile_openwebtext2"


class PilePhilPapers(PilePerplexityTask):
    DATASET_NAME = "pile_philpapers"


class PilePileCc(PilePerplexityTask):
    DATASET_NAME = "pile_pile-cc"


class PilePubmedAbstracts(PilePerplexityTask):
    DATASET_NAME = "pile_pubmed-abstracts"


class PilePubmedCentral(PilePerplexityTask):
    DATASET_NAME = "pile_pubmed-central"


class PileStackExchange(PilePerplexityTask):
    DATASET_NAME = "pile_stackexchange"


class PileUspto(PilePerplexityTask):
    DATASET_NAME = "pile_upsto"


class PileUbuntuIrc(PilePerplexityTask):
    DATASET_NAME = "pile_ubuntu-irc"


class PileWikipedia(PilePerplexityTask):
    DATASET_NAME = "pile_wikipedia"


class PileYoutubeSubtitles(PilePerplexityTask):
    DATASET_NAME = "pile_youtubesubtitles"