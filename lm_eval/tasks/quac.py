import json
import random
import os
from lm_eval.base import Dataset
from ..utils import sh


class QuAC(Dataset):    
    def __init__(self):
        super().__init__()

    def download(self):
        if not os.path.exists('data/quac'):
            sh("""
                mkdir -p data/quac 
                wget https://s3.amazonaws.com/my89public/quac/train_v0.2.json -O data/quac/train_v0.2.json
                wget https://s3.amazonaws.com/my89public/quac/val_v0.2.json -O data/quac/val_v0.2.json
                """)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        myjson = json.load(open('data/quac/train_v0.2.json'))['data']
        return self.load_doc(myjson)

    def validation_docs(self):
        myjson = json.load(open('data/quac/val_v0.2.json'))['data']    
        return self.load_doc(myjson)

    def test_docs(self):
        raise NotImplementedError("QuAC has no test docs.")
    
    def fewshot_description(self):
        desc = "TITLE: Title of the context passage - subtitle of the passage\nPARAGRAPH: Passage describing the relevant information for answering questions.\n\nQ: Text of a question.\n\nA: Answer to the question, based on the passage. If it cannot be answered based on the passage, write CANNOTANSWER"
        return desc

    def load_doc(self, myjson):
        docs = []
        for item in myjson:
            title = item['title'] + ' - ' + item['section_title']
            paragraph = item['paragraphs'][0]['context'].replace("CANNOTANSWER", "")
            qas = item['paragraphs'][0]['qas']
            qa_pairs = [(qa['question'], qa['answers'][0]['text']) for qa in qas]
            for (question, answer) in qa_pairs:
                doc = { 'title': title, 'paragraph': paragraph, 'question': question, 'answer': answer }
                docs.append(doc)  
        return docs
    
    def doc_to_text(self, doc, include_target=True):
        text = 'TITLE: ' + doc['title'] + '\n' + 'PARAGRAPH: ' + doc['paragraph'] + '\n\n' + 'Q: ' + doc['question'] + '\n\n' + 'A: '
        if include_target:
            text += doc['answer']
        return text

    def evaluate(self, docs, lm):
        pass
