import numpy as np


def process_docs(dataset):
    def _detokenize(text):
        text = text.replace(" '", "'")
        text = text.replace(" \n", "\n")
        text = text.replace("\n ", "\n")
        text = text.replace(" n't", "n't")
        text = text.replace("`` ", '"')
        text = text.replace("''", '"')
        # punctuation
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        return text

    def _process(doc):
        return {
            "article": _detokenize(doc["article"]),
            "options": [_detokenize(option) for option in doc["options"]],
        }

    return dataset.map(_process)


def process_results(doc, results):
    gold = ["A", "B", "C", "D"].index(doc["answers"])
    r4_1 = np.argmax(results) == gold  # r4_1 = accuracy
    ranks = sorted(results, reverse=True)
    r4_2 = (ranks.index(results[gold]) == 1) + r4_1
    mrr = 1.0 / (ranks.index(results[gold]) + 1)  # `+ 1` for index offset
    return {"r@1": r4_1, "r@2": r4_2, "mrr": mrr}
