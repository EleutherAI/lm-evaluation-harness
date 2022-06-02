"""
The FLORES-101 Evaluation Benchmark
for Low-Resource and Multilingual Machine Translation
https://arxiv.org/pdf/2106.03193.pdf

HuggingFace Dataset: https://huggingface.co/datasets/gsarti/flores_101
"""
from lm_eval.base import PerplexityTask

_CITATION = """
@inproceedings{flores101,
  title={The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},
  author={Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm\'{a}n, Francisco and Fan, Angela},
  journal={arXiv preprint arXiv:2106.03193},
  year={2021}
}
"""


class Flores101(PerplexityTask):
    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        prompt=None,
        save_examples=True,
    ):
        super().__init__(
            data_dir,
            cache_dir,
            download_mode,
            # True! We want to track the performance across different topics/domains
            save_examples=save_examples,
        )
        self.save_examples = save_examples

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["dev"]

    def doc_to_target(self, doc):
        """This is a null prompt task. We need to get the target from the doc."""
        return doc["sentence"]

    def process_results(self, doc, results):
        if self.save_examples:
            out, log = super().process_results(doc, results)
            log["topic"] = doc["topic"]
            log["domain"] = doc["domain"]
            return out, log
        else:
            return super().process_results(doc, results)


LANGS = [
    "afr",
    "amh",
    "ara",
    "hye",
    "asm",
    "ast",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "mya",
    "cat",
    "ceb",
    "zho_simpl",
    "zho_trad",
    "hrv",
    "ces",
    "dan",
    "nld",
    "eng",
    "est",
    "tgl",
    "fin",
    "fra",
    "ful",
    "glg",
    "lug",
    "kat",
    "deu",
    "ell",
    "guj",
    "hau",
    "heb",
    "hin",
    "hun",
    "isl",
    "ibo",
    "ind",
    "gle",
    "ita",
    "jpn",
    "jav",
    "kea",
    "kam",
    "kan",
    "kaz",
    "khm",
    "kor",
    "kir",
    "lao",
    "lav",
    "lin",
    "lit",
    "luo",
    "ltz",
    "mkd",
    "msa",
    "mal",
    "mlt",
    "mri",
    "mar",
    "mon",
    "npi",
    "nso",
    "nob",
    "nya",
    "oci",
    "ory",
    "orm",
    "pus",
    "fas",
    "pol",
    "por",
    "pan",
    "ron",
    "rus",
    "srp",
    "sna",
    "snd",
    "slk",
    "slv",
    "som",
    "ckb",
    "spa",
    "swh",
    "swe",
    "tgk",
    "tam",
    "tel",
    "tha",
    "tur",
    "ukr",
    "umb",
    "urd",
    "uzb",
    "vie",
    "cym",
    "wol",
    "xho",
    "yor",
    "zul",
]


def make_class(lang):
    class Flores101Lang(Flores101):
        DATASET_NAME = lang

    return Flores101Lang


def construct_tasks():
    tasks = {}
    for lang in LANGS:
        # Dynamically create a class for each language with a different `DATASET_NAME`
        tasks[f"gsarti/flores_101_{lang}"] = make_class(lang)
    return tasks
