from . import lambada
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
from lm_eval.utils import sh
from best_download import download_file
import json
from functools import partial
import os 

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
      f = f"data/lambada/lambada_test_{self.LANG}.jsonl"
      url = f"http://eaidata.bmk.sh/data/lambada_test_{self.LANG}.jsonl"
      try:
        if not os.path.exists():
          download_file(
              url, 
              f, 
              CHECKSUMS[self.LANG]
          )
      except:
        # fallback - for some reason best_download doesnt work all the time here
        sh(f"wget {url} -O {f}")
        sh(f'echo "{CHECKSUMS[self.LANG]}  {f}" | sha256sum --check')


    def validation_docs(self):
      with open(f"data/lambada/lambada_test_{self.LANG}.jsonl") as fh:
        for line in fh:
          yield json.loads(line)

class MultilingualLAMBADAEN(MultilingualLAMBADA):
  def __init__(self):
    super().__init__('en')

class MultilingualLAMBADAFR(MultilingualLAMBADA):
  def __init__(self):
    super().__init__('fr')

class MultilingualLAMBADADE(MultilingualLAMBADA):
  def __init__(self):
    super().__init__('de')

class MultilingualLAMBADAIT(MultilingualLAMBADA):
  def __init__(self):
    super().__init__('it')

class MultilingualLAMBADAES(MultilingualLAMBADA):
  def __init__(self):
    super().__init__('es')

LANG_CLASSES = [MultilingualLAMBADAEN, MultilingualLAMBADAFR, MultilingualLAMBADADE, MultilingualLAMBADAIT, MultilingualLAMBADAES]

def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"lambada_mt_{lang}"] = lang_class
    return tasks

