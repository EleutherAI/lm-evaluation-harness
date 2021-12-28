"""
Measuring Coding Challenge Competence With APPS
https://arxiv.org/pdf/2105.09938
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""

from ..utils import sh
from best_download import download_file
from datetime import datetime, date
from lm_eval.base import Task
from lm_eval.base import rf
from lm_eval.metrics import mean,perplexity
from pathlib import Path
from .reindent import run as run_reindent
from .testing_util import run_test
from tqdm import tqdm
import io
import json
import logging
import math
import numpy as np
import os
import pprint
import random
import sys
import time
import torch
import transformers
import xml.etree.ElementTree as ET

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

# helper functions 
def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def generate_prompt(test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    peeking = 0.0
    peek_frac = 0.5
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"

    _input += "\nANSWER:\n"

    if peeking > 0.0:
        # Need to do some peeking.

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        sample_sol = min(sols, key=len)
        sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        num_to_keep = int(len(sample_sol_token_ids) * peeking)
        sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        _input += tokenizer.decode(sample_sol_token_ids)

    else:
        sample_sol = None

    return _input, sample_sol

class Apps(Task):
    VERSION = 0
    DATASET_PATH = Path("data/apps")

    def download(self):
        if self.DATASET_PATH.exists():
            return
        Path.mkdir(self.DATASET_PATH)
        url = "https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz"
        checksum = "d98c0334a48031d65225b687af723376"
        tar_path = self.DATASET_PATH / "APPS.tar.gz"
        download_file(url, str(tar_path), checksum)
        sh("""
        cd data/apps/
        tar -xf APPS.tar.gz
        """)

    def load_docs(self,subset,split_percentage=0):
        prob_path = os.path.join(self.DATASET_PATH,"APPS",subset)
        if subset=='validation':
            prob_path = os.path.join(self.DATASET_PATH,"APPS","train")
        if split_percentage>0.0:
            total_len = len(os.listdir(prob_path))
            if subset=='train':
                problem_ids = sorted(os.listdir(prob_path))[:int(total_len*split_percentage)]
            if subset=='validation':
                problem_ids = sorted(os.listdir(prob_path))[int(total_len*(1-split_percentage)):]
        else:
            problem_ids = sorted(os.listdir(prob_path))
        for pid,problem_num in enumerate(problem_ids):
            test_case_path = os.path.join(prob_path,problem_num, "input_output.json")
            prompt_path = os.path.join(prob_path,problem_num, "question.txt")
            starter_path = os.path.join(prob_path,problem_num, "starter_code.py")
            solutions_path = os.path.join(prob_path,problem_num, "solutions.json")
            
            if not os.path.exists(starter_path):
                starter_path = None
                answer_type = "\nUse Standard Input format\n"
            else:
                answer_type = "\nUse Call-Based format\n"

            if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
                continue

            if os.path.exists(solutions_path):
                prompt_text, sample_sol = generate_prompt(test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
                out_doc = {}
                out_doc["answer_type"] = answer_type
                out_doc['prompt'] = prompt_text
                out_doc['sample_sol'] = sample_sol
                out_doc['prob_path'] = test_case_path
                with open(os.path.join(solutions_path), "r") as f:
                    sols = json.load(f)
                    out_doc['solutions'] = sols
            yield out_doc

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.load_docs('train',0.7)

    def validation_docs(self):
        return self.load_docs('validation',0.3)

    def test_docs(self):
        return self.load_docs('test')

    def fewshot_description(self):
        desc = "todo"
        return desc

    def doc_to_text(self, doc):
        return "\nQUESTION:\n{}\n{}\nANSWER:\n".format(doc["prompt"], doc["sample_sol"])
    
    def doc_to_target(self, doc):
        return min(doc['solutions'],key=len)

    def construct_requests(self, doc, ctx):
        conn_request = rf.greedy_until(ctx, ["<|endoftext|>"])
        return conn_request


    def process_results(self, doc, results):
        results = run_test(doc['prob_path'],results[0])
        results = [False if x<0 else x for x in results] 

        return {
            'acc': 100*np.mean(results)
        }

    def aggregation(self):
        return {
            'acc': mean
        }

    def higher_is_better(self):
        return {
            'acc': True
        }


