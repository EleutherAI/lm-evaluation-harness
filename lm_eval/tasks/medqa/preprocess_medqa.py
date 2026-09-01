def doc_to_text(doc) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    return f"Question: {doc['sent1']}\n{answers}Answer:"


def doc_to_target(doc) -> int:
    return doc["label"]

def doc_to_text_cot(doc) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    
    prompt = f"Question: {doc['sent1']}\n{answers}"
    prompt += "First, think step-by-step. End your response with exactly this format:\nFinal Answer: X\n(where X is A, B, C, or D)."
    return prompt

def doc_to_target_cot(doc) -> str:
    choices = ["A", "B", "C", "D"]
    return choices[int(doc["label"])]
