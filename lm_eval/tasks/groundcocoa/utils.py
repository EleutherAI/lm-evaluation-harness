import datasets
from datasets import Dataset
import pandas as pd


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    cocoa_dataset = [sample for sample in dataset]
    processed = []
    for doc in cocoa_dataset:
        question = doc['query']
        question = question + '\n\n Option A: ' + str(doc['Option A']) + '\n'
        question = question + '\n Option B: ' + str(doc['Option B']) + '\n'
        question = question + '\n Option C: ' + str(doc['Option C']) + '\n'
        question = question + '\n Option D: ' + str(doc['Option D']) + '\n'
        question = question + '\n Option E: ' + str(doc['Option E']) + '\n'
        out_doc = {
            "query": question,
            "choices": ['A', 'B', 'C', 'D', 'E'],
            "gold": doc["Answer"],
        }
        processed.append(out_doc)
    df = pd.DataFrame(processed)
    dataset = Dataset.from_pandas(df)
    return dataset
