import datasets

UNKNOWN_RESPONSES = [
                "Cannot answer",
                "Can't be determined",
                "Cannot be determined",
                "Can't answer",
                "Not known",
                "Not enough info",
                "Not enough information",
                "Not answerable",
                "Undetermined",
                "Unknown",
            ]

def doc_to_choice(doc):
    choices = [doc["ans0"], doc["ans1"], doc["ans2"]]
    current_unknown_answer = list(set(choices) & set(UNKNOWN_RESPONSES))
    choices.remove(current_unknown_answer[0])
    choices += UNKNOWN_RESPONSES
    return choices

def doc_to_targets(doc):
    label = doc["label"]
    choices = [doc["ans0"], doc["ans1"], doc["ans2"]]
    target_word = choices[label]
    if target_word in UNKNOWN_RESPONSES:
        targets = list(range(2,2+len(UNKNOWN_RESPONSES)+1))
    else:
        targets = [doc_to_choice(doc).index(target_word)]
    return targets

def filter_context_condition(dataset: datasets.Dataset, condition: str) -> datasets.Dataset:
    return dataset.filter(lambda example: example["context_condition"] == condition)

def filter_ambiguous(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_context_condition(dataset, "ambig")

def filter_disambiguated(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_context_condition(dataset, "disambig")
