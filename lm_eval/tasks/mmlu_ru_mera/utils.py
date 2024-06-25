import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        instruction = doc["instruction"]
        inputs = doc["inputs"]
        output = doc["outputs"]
        return {"query": instruction.format(**inputs), "label": output}

    return dataset.map(_process_doc)


def process_docs_continuation(dataset: datasets.Dataset) -> datasets.Dataset:
    option_keys = ["option_a", "option_b", "option_c", "option_d"]
    outputs = ["A", "B", "C", "D"]

    def _process_doc(doc):
        instruction = doc["instruction"]
        inputs = doc["inputs"]
        choices = [out + " " + inputs[key] for out, key in zip(outputs, option_keys)]
        output = choices[outputs.index(doc["outputs"])]
        return {
            "query": instruction.format(**inputs),
            "label": output,
            "choices": choices,
        }

    return dataset.map(_process_doc)
