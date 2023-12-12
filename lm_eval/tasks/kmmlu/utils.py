import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        instruction = (
            f"다음을 읽고 정답으로 알맞은 것을 고르시요.\n"
            f"### Question: {doc['question']}\n"
            f"### Options:\n"
            f"(1) {doc['option#1']}\n(2) {doc['option#2']}\n(3) {doc['option#3']}\n(4) {doc['option#4']}\n"
            f"### Answer: 주어진 문제의 정답은"
        )
        out_doc = {
            "question": instruction,
            "choices": ["(1)", "(2)", "(3)", "(4)"],
            "gold": int(doc["answer"]) - 1,
        }
        return out_doc

    return dataset.map(_process_doc)
