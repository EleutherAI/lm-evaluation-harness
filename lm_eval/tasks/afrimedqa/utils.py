import datasets

import json

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    OPTIONS = ['option1', 'option2', 'option3', 'option4', 'option5']

    def _filter_doc(doc):
        is_expert = doc['tier'] == 'expert'
        has_correct_answer = doc["correct_answer"] and not ',' in doc["correct_answer"]
        has_correct_option = doc["answer_options"] and len(json.loads(doc["answer_options"])) == len(OPTIONS)
        return doc["question"] and is_expert and has_correct_answer and has_correct_option
    
    def _process_doc(doc):
        out_doc = {
            "question": doc["question"],
            "choices": [json.loads(doc["answer_options"])[opt] for opt in OPTIONS],
            "gold": OPTIONS.index(doc["correct_answer"]),
        }
        return out_doc

    return dataset.filter(_filter_doc).map(_process_doc)
