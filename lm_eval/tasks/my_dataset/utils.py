# File: lm_eval/tasks/halftruthdetection/utils.py
import datasets
# import huggingface

LABEL_MAPPING = {
    "true": 'A',
    "mostly-true": 'B',
    "half-true": 'C',
    "mostly-false": 'D',
    "false": 'E',
    "pants-fire": 'F'
}

def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = ['A', 'B', 'C', 'D', 'E', 'F']
        doc["gold"] = LABEL_MAPPING[doc["label"]]
        return doc

    return dataset.map(_helper)

def get_choices(doc):
    return doc["choices"]

def get_target(doc):
    return doc["gold"]
