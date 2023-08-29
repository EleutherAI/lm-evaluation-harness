import datasets


def _doc_to_text_all(doc: dict) -> str:
    all_choices = " ".join(doc["options"])
    passage = doc.get("passage", None)
    if passage is None:
        return f"Problem: {doc['question']}\n{all_choices}\nAnswer:"
    return f"Problem: {passage}\n{doc['question']}\n{all_choices}\nAnswer:"


# taken from
# https://github.com/microsoft/AGIEval/blob/19b2c5daed87e3463fe6a29f0c342bfc31e98234/src/dataset_loader.py#L25
# Used for all AGI datasets except MATH.
# Not used here yet!
def doc_to_text_zeroshot(doc: dict) -> str:
    # No space after passage!
    """
    <passage>Q: <question> Answer Choices: <choice1> <choice2> <choice3> <choice4>\n
    A: Among A through {A-D}, the answer is
    """
    passage = doc.get("passage", "")
    option_string = "ABCDEFG"
    count = len(doc["options"])
    if count == 1:
        count = 5
    return (
        passage
        + "Q: "
        + doc["question"]
        + " "
        + "Answer Choices: "
        + " ".join(doc["options"])
        + "\n"
        + "A: Among A through {}, the answer is".format(option_string[count - 1])
    )
