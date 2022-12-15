"""
German Europarl NER data (PPL only)
http://www.nlpado.de/~sebastian/pub/papers/konvens10_faruqui.pdf

We present a freely available optimized Named Entity Recognizer (NER) for German.
It alleviates the small size of available NER training corpora for German with
distributional generalization features trained on large unlabelled corpora.
e vary the size and source of the generalization corpus and find improvements
of 6% F1-score (in-domain) and 9% (out-of-domain) over simple supervised training.

Dataset: https://nlpado.de/~sebastian/software/ner_german.shtml

NOTE: This dataset is used as language modeling tasks (perplexity) and NOT named entity recogniton.

"""
import re
import datasets
from lm_eval.base import PerplexityTask


_CITATION = """
@InProceedings{faruqui10:_training
  author =       {Manaal Faruqui and Sebastian Pad\'{o}},
  title =        {Training and Evaluating a German Named Entity Recognizer
                  with Semantic Generalization},
  booktitle = {Proceedings of KONVENS 2010},
  year =         2010,
  address =      {Saarbr\"ucken, Germany}}
}
"""


class GermanEuroparlPerplexity(PerplexityTask):
    VERSION = 0
    text_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        europarl_ner_german_test_path = (
            "https://nlpado.de/~sebastian/software/ner/ep-96-04-15.conll"
        )

        self.dataset = datasets.load_dataset(
            "text",
            data_files={
                "test": europarl_ner_german_test_path,
            },
            encoding="windows-1252",
        )

        # Convert CONLL to plain text
        self.text_docs = (
            " ".join(
                [
                    (line.split()[0] if len(line) > 0 else "\n")
                    for line in self.dataset["test"]["text"]
                ]
            )
        ).splitlines()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.text_docs

    def doc_to_target(self, doc):
        return doc

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in doc
        return len(re.split(r"\s+", doc))
