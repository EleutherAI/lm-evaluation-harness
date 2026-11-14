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
    ranked_choices = sorted(
        range(len(results)), key=lambda idx: results[idx][0], reverse=True
    )
    rank = ranked_choices.index(gold)
    return {"r@1": rank == 0, "r@2": rank < 2, "mrr": 1.0 / (rank + 1)}
