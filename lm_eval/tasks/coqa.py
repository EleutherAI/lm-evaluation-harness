import json
import random
from lm_eval.base import Dataset
from . import TASK_REGISTRY


@TASK_REGISTRY.register("coqa")
class CoQA(Dataset):
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        myjson = json.load(open('data/coqa/coqa-train-v1.0.json'))['data']
        return self.load_doc(myjson)

    def validation_docs(self):
        pass

    def test_docs(self):
        myjson = json.load(open('data/coqa/coqa-dev-v1.0.json'))['data']    
        return self.load_doc(myjson)
    
    def fewshot_examples(self, k):
        traindocs = list(self.training_docs())
        random.seed(123)
        random.shuffle(traindocs)

        return traindocs[:k]
    
    def fewshot_description(self):
        pass

    def load_doc(self, myjson):
        docs = []
        for item in myjson:
            new_instance = [item['story']]
            qa_pairs = zip(item['questions'], item['answers'])
            for pair in qa_pairs:
                new_instance.append('\n')
                new_instance.append(''.join(['Q: ',pair[0]['input_text']]))
                new_instance.append(''.join(['A: ',pair[1]['input_text']]))
            docs.append(new_instance)  
        return docs
    
    def doc_to_text(self, doc, include_target=True):
        text = '\n<|endoftext|>\n'.join(['\n'.join(instance) for instance in doc])
        text = text + '\n<|endoftext|>'
        return text

    def evaluate(self, docs, lm):
        pass
