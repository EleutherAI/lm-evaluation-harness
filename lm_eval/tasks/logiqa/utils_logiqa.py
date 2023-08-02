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
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def doc_to_target(doc) -> int:
    choices = ["a", "b", "c", "d"]
    return choices.index(doc["label"].strip())
