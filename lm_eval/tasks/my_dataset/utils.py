# File: lm_eval/tasks/halftruthdetection/utils.py
import datasets
# import huggingface

# LABEL_MAPPING = {
#     "true": 'A',
#     "mostly-true": 'B',
#     "half-true": 'C',
#     "mostly-false": 'D',
#     "false": 'E',
# }

def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = ['true', 'mostly-true', 'half-true', 'mostly-false', 'false']
        return doc

    return dataset.map(_helper)

def get_choices(doc):
    return doc["choices"]

# def get_target(doc):
#     return doc["gold"]
