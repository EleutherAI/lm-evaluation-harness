import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
                
        gold = doc["choices"].index(doc["answer"])
        
        instruction = f"""다음을 읽고 정답으로 알맞은 것을 고르시요.
### Question: {doc["question"]}
### Options:
(1) {doc["choices"][0]}\n(2) {doc["choices"][1]}\n(3) {doc["choices"][2]}\n(4) {doc["choices"][3]}
### Answer: 주어진 문제의 정답은"""

        out_doc = {
            "question": instruction,
            "choices": ["(1)", "(2)", "(3)", "(4)", "(5)"],
            "gold": gold,
        }
        return out_doc

    return dataset.map(_process_doc)
