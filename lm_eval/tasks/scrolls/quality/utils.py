import re
from functools import partial

from ..preprocessors import process_docs_prepended_question

def process_docs(dataset):

    dataset = process_docs_prepended_question(dataset)

    _multiple_choice_pattern = re.compile(r" *\([A-D]\) *")

    def _normalize_answer(text):
        return " ".join(text.split()).strip()

    def _process_doc(doc):

        split = doc["text"].find("\n\n", doc["text"].find("(D)"))
        choices_text = doc["text"][:split]

        doc["text"] = doc["text"][split:].strip()
        doc["choices"] = [_normalize_answer(choice) for choice in re.split(
            _multiple_choice_pattern, choices_text)[1:]]
        doc["gold"] = doc["choices"].index(_normalize_answer(doc["outputs"][0]))

        return doc

    return dataset.map(_process_doc)
