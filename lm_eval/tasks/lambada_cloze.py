import json
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
from lm_eval.utils import sh
from lm_eval.tasks.lambada import LAMBADA
from best_download import download_file


class LAMBADA_cloze(LAMBADA):
    VERSION = 0
    def doc_to_text(self, doc):
        return doc['text'].rsplit(' ', 1)[0] + " ____. ->"

    def doc_to_target(self, doc):
        return " " + doc['text'].rsplit(' ', 1)[1]
    
    def fewshot_description(self):
        return "Fill in blank:\n"
