
from ...base import Dataset
import os
import json
class CoQA(Dataset):
    def training_docs(self):
        pass

    def validation_docs(self):
        pass
    
    def test_docs(self):
        pass
    
    def fewshot_examples(self, k):
        traindocs = list(self.training_docs())
        random.seed(123)
        random.shuffle(traindocs)

        return traindocs[:k]
    
    def fewshot_description(self):
        pass

    def doc_to_text(self, doc, include_target=True):
        json.load(open(doc))

        
 
