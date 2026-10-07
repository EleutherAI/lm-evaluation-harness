import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # Each file is one document of blank-line separated records: the answer
    # letter, the passage, the question, then one "X." prefixed line per option.
    # Same parsing as the former EleutherAI/logiqa loading script.
    def normalize(text):
        return text.replace(".", ". ").strip()

    docs = []
    for doc in dataset:
        for record in doc["text"].strip().split("\n\n"):
            lines = record.split("\n")
            docs.append(
                {
                    "label": lines[0].strip(),
                    "context": normalize(lines[1]),
                    "question": normalize(lines[2]),
                    "options": [normalize(option[2:]) for option in lines[3:]],
                }
            )
    return datasets.Dataset.from_list(docs)


# Copied from Master
def doc_to_text(doc) -> str:
    """
    Passage: <passage>
    Question: <question>
    Choices:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    choices = ["a", "b", "c", "d"]
    prompt = "Passage: " + doc["context"] + "\n"
    prompt += "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in zip(choices, doc["options"], strict=False):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def doc_to_target(doc) -> int:
    choices = ["a", "b", "c", "d"]
    return choices.index(doc["label"].strip())
