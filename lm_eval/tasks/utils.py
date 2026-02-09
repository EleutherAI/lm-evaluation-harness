from typing import Dict, Any
import datasets


def process_docs(dataset):
    def _process_doc(doc):
        options = doc["options"].strip('[]').replace("'", "").split(', ')

        return {
            "question": str(doc["question"]).strip(),
            "options": ', '.join(options),
            "correct_option": str(doc["correct_option"]).strip().lower(),
        }
    return dataset.map(_process_doc)
