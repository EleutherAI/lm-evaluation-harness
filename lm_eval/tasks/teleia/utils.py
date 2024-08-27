import re

import datasets


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_cervantes(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        # breakpoint()
        out_doc = {
            "query": "Pregunta: " + preprocess(doc["question"]) + "\nRespuesta:",
            "choices": [
                preprocess(option)
                for option in [
                    doc["option_a"],
                    doc["option_b"],
                    doc["option_c"],
                    doc["option_d"]
                ]
                if option
            ],
            "target": ["A", "B", "C", "D"].index(doc["answer"]),
        }
        return out_doc

    return dataset.map(_process_doc)
    

def process_pce_siele(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        # breakpoint()
        out_doc = {
            "query": "Pregunta: " + preprocess(doc["question"]) + "\nRespuesta:",
            "choices": [
                preprocess(option)
                for option in [
                    doc["option_a"],
                    doc["option_b"],
                    doc["option_c"],
                ]
                if option
            ],
            "target": ["A", "B", "C"].index(doc["answer"]),
        }
        return out_doc

    return dataset.map(_process_doc)
