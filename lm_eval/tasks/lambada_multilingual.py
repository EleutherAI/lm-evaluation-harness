from . import lambada
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
from lm_eval.utils import sh
from best_download import download_file
import json

# This task is lambada but machine-translated to the other languages.

LANGS = ["en", "fr", "de", "it", "es"]
CHECKSUMS = {"en": "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226", 
             "fr": "941ec6a73dba7dc91c860bf493eb66a527cd430148827a4753a4535a046bf362", 
             "de": "51c6c1795894c46e88e4c104b5667f488efe79081fb34d746b82b8caa663865e", 
             "it": "86654237716702ab74f42855ae5a78455c1b0e50054a4593fb9c6fcf7fad0850", 
             "es": "ffd760026c647fb43c67ce1bc56fd527937304b348712dce33190ea6caba6f9c"
            }

class MultilingualLAMBADA(lambada.LAMBADA):
    VERSION = 0
    
    def __init__(self, lang=None):
      self.LANG = lang
      super().__init__()
    
    def download(self):
      sh("mkdir -p data/lambada")
      download_file(
          f"http://eaidata.bmk.sh/data/lambada_test_{self.LANG}.jsonl", 
          f"data/lambada/lambada_test_{self.LANG}.jsonl", 
          CHECKSUMS[self.LANG]
      )

    def validation_docs(self):
      with open(f"data/lambada/lambada_test_{self.LANG}.jsonl") as fh:
        for line in fh:
          yield json.loads(line)

          
def construct_tasks():
    tasks = {}
    for lang in LANGS:
        tasks[f"lambada_mt_{lang}"] = partial(MultilingualLAMBADA, lang=lang)
    return tasks
