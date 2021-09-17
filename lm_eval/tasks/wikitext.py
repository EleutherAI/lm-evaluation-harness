import os
import re
from lm_eval.base import rf, PerplexityTask
from lm_eval.utils import sh

from best_download import download_file


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


class WikiText(PerplexityTask):
    VERSION = 0

    def download(self):
        if not os.path.exists('data/wikitext/wikitext-2-raw/wiki.valid.raw'):
            os.makedirs("data/wikitext/", exist_ok=True)
            download_file("https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip", "data/wikitext/wikitext-2-raw-v1.zip", "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11")
            sh("cd data/wikitext/ && unzip wikitext-2-raw-v1.zip")

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def has_validation_docs(self):
        return True

    def has_train_docs(self):
        return True

    def has_test_docs(self):
        return True
    
    def docs_for_split(self, split):
        ret = []
        for line in open(f"data/wikitext/wikitext-2-raw/wiki.{split}.raw").read().split('\n'):
            rline = line.replace("= = =", "===").replace("= =", "==").strip()
            if rline.startswith('= ') and rline.strip().endswith(' ='):
                s = '\n'.join(ret)
                if s.strip(): yield s
                ret = []
            ret.append(line)
        yield '\n'.join(ret)

    def validation_docs(self):
        return self.docs_for_split('valid')

    def train_docs(self):
        return self.docs_for_split('train')

    def test_docs(self):
        return self.docs_for_split('test')

    def doc_to_target(self, doc):
        return wikitext_detokenizer(doc)
    
    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))