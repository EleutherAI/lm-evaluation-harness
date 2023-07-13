# Copied from Master
def doc_to_text(doc) -> str:
    """
    Passage: <passage>
    Question: <question>
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    choices = ["a", "b", "c", "d"]
    prompt = "Passage: " + doc["context"] + "\n"
    prompt += "Question: " + doc["question"] + "\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def doc_to_target(doc) -> str:
    choices = {"a": 0, "b": 1, "c": 2, "d": 3}
    label = choices[doc["label"]]
    return doc["options"][label]


def gold(doc) -> int:
    choices = {"a": 0, "b": 1, "c": 2, "d": 3}
    return choices[doc["label"]]
