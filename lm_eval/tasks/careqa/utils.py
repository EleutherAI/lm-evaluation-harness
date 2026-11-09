def doc_to_text(doc) -> str:
    """
    Question: <question>
    Choices:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    if doc["question"] is None:
        doc = {
            "question": "In relation to the immune mechanism involved in the rejection of transplanted solid organs, indicate the incorrect answer:",
            "op1": "Acute T-cell mediated rejection can be controlled through the use of drugs such as cyclosporine A or corticosteroids.",
            "exam_id": 36,
            "op3": "Chronic rejection or chronic graft injury is associated with endothelial damage mediated by anti-HLA antibodies.",
            "category": "Medicine",
            "unique_id": "5636d1af-e0b1-43b0-8a04-6f127dcf6785",
            "op4": "Hyperacute rejection is mediated by cytotoxic T lymphocytes against donor antigens present in the recipient.",
            "op2": "The presence of specific antibodies against the donor (DSA) in the recipient prior to transplantation is a contraindication for it.",
            "cop": 4,
            "year": 2024,
        }
    choices = [doc["op1"], doc["op2"], doc["op3"], doc["op4"]]
    option_choices = {
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
    }

    prompt = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def doc_to_target(doc) -> int:
    return doc["cop"] - 1
