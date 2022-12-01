"""
OSCAR
https://oscar-project.org/


OSCAR or Open Super-large Crawled Aggregated coRpus is a huge multilingual
corpus obtained by language classification and filtering of the Common Crawl
corpus using the Ungoliant architecture.

Dataset:
- Original https://huggingface.co/datasets/oscar
- Subset https://huggingface.co/datasets/malteos/wechsel_de

NOTE: This only contains the German validation subset used by WECHSEL ()

"""
import re
import datasets
from lm_eval.base import PerplexityTask


_CITATION = """
@inproceedings{ortiz-suarez-etal-2020-monolingual,
    title = "A Monolingual Approach to Contextualized Word Embeddings for Mid-Resource Languages",
    author = "Ortiz Su{'a}rez, Pedro Javier  and
      Romary, Laurent  and
      Sagot, Benoit",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.156",
    pages = "1703--1714",
    abstract = "We use the multilingual OSCAR corpus, extracted from Common Crawl via language classification, filtering and cleaning, to train monolingual contextualized word embeddings (ELMo) for five mid-resource languages. We then compare the performance of OSCAR-based and Wikipedia-based ELMo embeddings for these languages on the part-of-speech tagging and parsing tasks. We show that, despite the noise in the Common-Crawl-based OSCAR data, embeddings trained on OSCAR perform much better than monolingual embeddings trained on Wikipedia. They actually equal or improve the current state of the art in tagging and parsing for all five languages. In particular, they also improve over multilingual Wikipedia-based contextual embeddings (multilingual BERT), which almost always constitutes the previous state of the art, thereby showing that the benefit of a larger, more diverse corpus surpasses the cross-lingual benefit of multilingual embedding architectures.",
}
"""


class OscarPerplexityGerman(PerplexityTask):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = datasets.load_dataset(
            "malteos/wechsel_de",
            data_files={
                "test": "valid.random_1636.json.gz",
            },
        )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return doc

    def count_words(self, doc):
        # count number of words in doc
        return len(re.split(r"\s+", doc))
