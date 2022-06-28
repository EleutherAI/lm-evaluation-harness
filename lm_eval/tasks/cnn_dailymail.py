"""
CNN/Daily Mail is a dataset for text summarization. Human generated abstractive
summary bullets were generated from news stories in CNN and Daily Mail websites
as questions (with one of the entities hidden), and stories as the corresponding
passages from which the system is expected to answer the fill-in the-blank
question. The authors released the scripts that crawl, extract and generate pairs
of passages and questions from these websites.

In all, the corpus has 286,817 training pairs, 13,368 validation pairs and 11,487
test pairs, as defined by their scripts. The source documents in the training set
have 766 words spanning 29.74 sentences on an average while the summaries consist
of 53 words and 3.72 sentences. """
from lm_eval.api.task import PromptSourceTask


_CITATION = """@article{DBLP:journals/corr/NallapatiXZ16,
  author    = {Ramesh Nallapati and
               Bing Xiang and
               Bowen Zhou},
  title     = {Sequence-to-Sequence RNNs for Text Summarization},
  journal   = {CoRR},
  volume    = {abs/1602.06023},
  year      = {2016},
  url       = {http://arxiv.org/abs/1602.06023},
  eprinttype = {arXiv},
  eprint    = {1602.06023},
  timestamp = {Mon, 13 Aug 2018 16:46:52 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/NallapatiXZ16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}"""


class CnnDailyMail(PromptSourceTask):

    DATASET_PATH = "cnn_dailymail"
    DATASET_NAME = "3.0.0"

    def doc_to_rawtext(self, doc):
        return doc["article"]

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def max_generation_length(self):
        return 200
