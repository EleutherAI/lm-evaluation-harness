from functools import partial


# Official zero-shot instruction from the SuperGPQA reference implementation:
# https://github.com/SuperGPQA/SuperGPQA/blob/main/config/prompt/zero-shot.yaml
INSTRUCTION = (
    "Answer the following multiple choice question. There is only one correct answer. "
    "The last line of your response should be in the format 'Answer: $LETTER' "
    "(without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.\n\n"
)


def doc_to_text(doc) -> str:
    options = "\n".join(
        f"{chr(ord('A') + i)}) {option}" for i, option in enumerate(doc["options"])
    )
    return INSTRUCTION + doc["question"] + "\n" + options


def process_docs(dataset, discipline):
    return dataset.filter(lambda x: x["discipline"] == discipline)


process_agronomy = partial(process_docs, discipline="Agronomy")
process_economics = partial(process_docs, discipline="Economics")
process_education = partial(process_docs, discipline="Education")
process_engineering = partial(process_docs, discipline="Engineering")
process_history = partial(process_docs, discipline="History")
process_law = partial(process_docs, discipline="Law")
process_literature_and_arts = partial(process_docs, discipline="Literature and Arts")
process_management = partial(process_docs, discipline="Management")
process_medicine = partial(process_docs, discipline="Medicine")
process_military_science = partial(process_docs, discipline="Military Science")
process_philosophy = partial(process_docs, discipline="Philosophy")
process_science = partial(process_docs, discipline="Science")
process_sociology = partial(process_docs, discipline="Sociology")
