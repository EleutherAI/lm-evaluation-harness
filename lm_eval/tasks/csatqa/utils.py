import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        instruction = f"""다음을 읽고 정답으로 알맞은 것을 고르시요.
### Context: {doc["context"]}
### Question: {doc["question"]}
### Options:
(1) {doc["option#1"]}\n(2) {doc["option#2"]}\n(3) {doc["option#3"]}\n(4) {doc["option#4"]}\n(5) {doc["option#5"]}
### Answer: 주어진 문제의 정답은"""

        out_doc = {
            "question": instruction,
            "choices": ["(1)", "(2)", "(3)", "(4)", "(5)"],
            "gold": int(doc["gold"]) - 1,
        }
        return out_doc

    return dataset.map(_process_doc)
