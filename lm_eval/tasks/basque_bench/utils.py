from functools import partial


# ~~~~~~~~~~~ XCOPA ~~~~~~~~~~~ #

xcopa_connectors = {"cause": " Izan ere,", "effect": " Beraz,"}


def xcopa_doc_to_text(doc):
    conn = xcopa_connectors[doc["question"]]
    return doc["premise"].strip() + f"{conn}"


def xcopa_doc_to_choice(doc):
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]

    return [convert_choice(doc["choice1"]), convert_choice(doc["choice2"])]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
