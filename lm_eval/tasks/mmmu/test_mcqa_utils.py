import ast


def process_docs(dataset):
    return dataset.filter(lambda x: x["question_type"] == "multiple-choice")


def doc_to_choice(doc):
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I"][
        : len(ast.literal_eval(doc["options"]))
    ]


def doc_to_target(doc):
    print(doc)
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I"].index(doc["answer"])
