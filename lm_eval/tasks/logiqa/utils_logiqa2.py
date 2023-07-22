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
    prompt = "Passage: " + doc["text"] + "\n"
    prompt += "Question: " + doc["question"] + "\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


# # https://github.com/csitfun/LogiQA2.0/blob/main/logiqa2nli/nli-prompt.py
# def doc_to_textNLI(doc):
#     maj_premise = ' '.join(list(doc['major_premise']))
#     min_premise = ' '.join(list(doc['minor_premise']))
#     hypo = doc['conclusion']
#     prompt_input = "Given the fact: " + maj_premise + ' ' + min_premise + " Does it follow that: " + hypo + " Yes or no?"
#     return prompt_input
