import numpy as np
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from . common import NLP_TASK, simple_accuracy_metric, yesno
from pathlib import Path
from ..base import Dataset

class DROP(Dataset):
    DATAFOLDER = Path(__file__).parent / "../../data/drop"
    
    def has_training_docs(self):
        """Whether the task has a training set"""
        return True
    
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return True

    def has_test_docs(self):
        """Whether the task has a test set"""
        return False

    def training_docs(self):
        docs = json.load(open(self.DATAFOLDER / 'drop_dataset_train.json'))
        return [docs[k] for k in docs.keys()]


    def validation_docs(self):
        docs = json.load(open(self.DATAFOLDER / 'drop_dataset_dev.json'))
        return [docs[k] for k in docs.keys()]
    
    def test_docs(self):
        pass
    
    def doc_to_text(self, doc, include_target=True):
        doctext = "Passage: {}\n\n".format(doc["passage"])
        qa_texts = []
        for pair in doc["qa_pairs"]:
            text = ''.join(['Q: ', pair['question'],'\nA: '])
            if include_target:
                def get_answer(ans_dict):
                    if ans_dict['number'] != '':
                        return ans_dict['number']
                    if ans_dict['spans'] != []:
                        if len(ans_dict['spans']) > 0:
                            return ', '.join(ans_dict['spans'])
                        return ans_dict['spans'][0]
                    return ' '.join([ans_dict['date']['day'], 
                                     ans_dict['date']['month'], 
                                     ans_dict['date']['year']]).strip() 
                text = ''.join([text, get_answer(pair['answer'])])
            qa_texts.append(text)
        return ''.join([doctext, '\n\n'.join(qa_texts)])
            
    
    def evaluate(self, docs, lm, provide_description, num_fewshot):
        """Take iterable of docs and evaluates, returning a dict with the following format:

        {
            "major": float,
            "minor": dict,
            "higher_is_better": bool,
        }

        * `major` should be a single, representative number, for programmatic comparison
        * `minor` should be a dictionary containing all relevant sub-metrics
        * `higher_is_better` determines whether a higher metric is better
        """
        pass

    def fewshot_description(self):
        return "Read the passage and answer the questions "

