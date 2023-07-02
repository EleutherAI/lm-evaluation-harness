def partial_context(doc, option):
    # Substitute the pronoun in the sentence with the specified option
    # and ignore everything after.
    pronoun_loc = doc["sentence"].index("_")
    return doc["sentence"][:pronoun_loc] + option

def partial_target(doc):
    # The target is everything after the document specified pronoun.
    pronoun_loc = doc["sentence"].index("_") + 1
    return doc["sentence"][pronoun_loc:].strip()

def create_choices(doc):
    choices = []
    for option in [doc["option1"], doc["option2"]]:
        partial_ctx = partial_context(doc, option)
        choices.append(partial_ctx)
    return choices

def gold_alias(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc['answer']]