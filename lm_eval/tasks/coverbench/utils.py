from functools import partial


def doc_to_target(example):
    target = example["label"]
    if target == "true":
        return "Correct"
    return "Incorrect"


def doc_to_text(example, cot=False):
    context = example["context"]
    claim = example["claim"]
    prompt = f"Your task is to check if the Claim is correct according to the Evidence. Generate ’Correct’ if the Claim is correct according to the Evidence, or ’Incorrect’ if the claim is incorrect or cannot be verified.\n\nEvidence: {context}\n\nClaim: {claim}\n\n"
    if cot:
        prompt += "Let’s think step-by-step:"
    else:
        prompt += "Answer:"
    return prompt


doc_to_text_zeroshot = partial(doc_to_text, cot=False)
doc_to_text_zeroshot_cot = partial(doc_to_text, cot=True)
