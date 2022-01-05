"""
MuTual: A Dataset for Multi-Turn Dialogue Reasoning
https://www.aclweb.org/anthology/2020.acl-main.130/

@inproceedings{mutual,
    title = "MuTual: A Dataset for Multi-Turn Dialogue Reasoning",
    author = "Cui, Leyang  and Wu, Yu and Liu, Shujie and Zhang, Yue and Zhou, Ming" ,
    booktitle = "Proceedings of the 58th Conference of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
"""
import json
import zipfile
import shutil
import numpy as np
from pathlib import Path
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from best_download import download_file


class MuTualBase(Task):
    VERSION = 1
    BASE_PATH = Path("data/mutual")
    DATASET_NAME = None
    CHOICES = ['A', 'B', 'C', 'D']

    def __init__(self):
        super().__init__()

    def download(self):
        if self.BASE_PATH.exists():
            return
        Path.mkdir(self.BASE_PATH, parents=True)
        master_zip = Path("data/master.zip")
        download_file(
            "https://github.com/Nealcly/MuTual/archive/master.zip",
            str(master_zip),
            "bb325cf6c672f0f02699993a37138b0fa0af6fcfc77ec81dfbe46add4d7b29f9")
        with zipfile.ZipFile(master_zip, 'r') as zip:
            zip.extractall("data")
        Path("data/MuTual-master/data").rename(str(self.BASE_PATH))
        # Remove left over files and directories.
        master_zip.unlink()
        shutil.rmtree("data/MuTual-master")

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def _load_docs(self, path):
        for file in sorted(path.iterdir()):
            if file.suffix != ".txt":
                continue
            with open(file, 'r', encoding='utf-8') as f:
                yield json.load(f)

    def training_docs(self):
        return self._load_docs(self.BASE_PATH / self.DATASET_NAME / "train")

    def validation_docs(self):
        return self._load_docs(self.BASE_PATH / self.DATASET_NAME / "dev")

    def test_docs(self):
        return NotImplemented

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def doc_to_text(self, doc):
        return self.detokenize(doc["article"])

    def doc_to_target(self, doc):
        return " " + self.detokenize(doc["options"][self.CHOICES.index(doc["answers"])])

    def construct_requests(self, doc, ctx):
        lls = []
        for option in doc["options"]:
            lls.append(rf.loglikelihood(ctx, f" {self.detokenize(option)}")[0])
        return lls

    def detokenize(self, text):
        text = text.replace(" '", "'")
        text = text.replace(" \n", "\n")
        text = text.replace("\n ", "\n")
        text = text.replace(" n't", "n't")
        text = text.replace("`` ", '"')
        text = text.replace("''", '"')
        # punctuation
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        return text

    def process_results(self, doc, results):
        gold = self.CHOICES.index(doc["answers"])
        r4_1 = np.argmax(results) == gold  # r4_1 = accuracy
        ranks = sorted(results, reverse=True)
        r4_2 = (ranks.index(results[gold]) == 1) + r4_1
        mrr = 1. / (ranks.index(results[gold]) + 1)  # `+ 1` for index offset
        return {
            "r@1": r4_1,
            "r@2": r4_2,
            "mrr": mrr
        }

    def aggregation(self):
        return {
            "r@1": mean,
            "r@2": mean,
            "mrr": mean
        }

    def higher_is_better(self):
        return {
            "r@1": True,
            "r@2": True,
            "mrr": True
        }


class MuTual(MuTualBase):
    DATASET_NAME = Path("mutual")


class MuTualPlus(MuTualBase):
    DATASET_NAME = Path("mutual_plus")
