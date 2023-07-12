def convert_choice(choice):
    return choice[0].lower() + choice[1:]


def doc_to_text(doc):
    # Drop the period
    connector = {
        "cause": "because",
        "effect": "therefore",
    }[doc["question"]]
    return doc["premise"].strip()[:-1] + f" {connector}"


def doc_to_target(doc):
    correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
    # Connect the sentences
    return " " + convert_choice(correct_choice)


def doc_to_choice(doc):
    return [" " + convert_choice(doc["choice1"]), " " + convert_choice(doc["choice2"])]
