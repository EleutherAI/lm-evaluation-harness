import pandas as pd
from datasets import Dataset

def process_docs(dataset: Dataset):
    def _helper(doc):
        doc["choices"] = ['A', 'B']
        doc["gold"] = 'A' if doc["Stereotype"]=='No' else 'B'
        return doc
    
    return dataset.map(_helper)