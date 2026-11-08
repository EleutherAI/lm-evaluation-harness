def xcopa_doc_to_choice(doc):
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]

    return [convert_choice(doc["choice1"]), convert_choice(doc["choice2"])]
