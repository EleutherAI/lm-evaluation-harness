import datasets


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = {
            "label": [item["marker"] for item in doc["answers"]],
            "text": [item["text"] for item in doc["answers"]],
        }
        del doc["answers"]
        return doc

    return dataset.map(_helper)


def p1(doc):
    prompt = "Питання: {question}\n{choices}\n\nПравильна відповідь: А, Б, В чи Г?\nВiдповiдь:"
    choices = "\n".join(
        [
            f"{key}. {value}"
            for key, value in zip(
                doc["choices"]["label"], doc["choices"]["text"], strict=False
            )
        ]
    )
    return prompt.format(question=doc["question"], choices=choices)


def p2(doc):
    prompt = "Питання: {question}\n\nВиберіть відповідь зі списку варіантів: {choices}\n\nВiдповiдь:"
    choices = "".join([f"\n- {choice}" for choice in doc["choices"]["text"]])
    return prompt.format(question=doc["question"], choices=choices)
