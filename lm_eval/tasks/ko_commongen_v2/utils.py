import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        query = f"""concept set: {{{doc['concept_set'].replace("#", ", ")}}}\n"""
        query += "\n".join([f"{i+1}. {doc[str(i+1)]}" for i in range(4)])

        out_doc = {
            "query": query,
            "choices": [f"{i+1}. {doc[str(i+1)]}" for i in range(4)],
            "gold": doc["gold"]
            - 1,  # The integer used to index into the correct element of `"choices"`.
        }
        return out_doc

    return dataset.map(_process_doc)
