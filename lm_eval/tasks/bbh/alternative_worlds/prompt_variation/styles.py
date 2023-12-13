import re
import string

yes_no = ["Yes", "No"]


def parse_choices(doc):

    input_text = doc["input"]
    choice_string = input_text.split("Options:")
    if len(choice_string) == 2:
        choice_string = choice_string[-1]
        if ("- Yes" in choice_string) and ("- No" in choice_string):
            choices = yes_no
        else:
            choices = [
                c[4:].rstrip("\n")
                for c in re.findall(r"\([A-Z]\) .*?\n|\([A-Z]\) .*?$", choice_string)
            ]
        return choices
    else:
        return []


def styles_01(doc):
    # Check for choices and remove them
    choices = parse_choices(doc)
    if choices != []:
        doc_to_text = doc["input"].split("Options:")[0]
        if doc_to_text[-1] in ["\n", " "]:
            doc_to_text = doc_to_text[:-1]
    else:
        doc_to_text = doc["input"]
    return doc_to_text


def styles_02(doc):
    # Check for choices and remove them
    doc_to_text = styles_01(doc)
    return "Q: " + doc_to_text + "\nA:"


def styles_03(doc):
    # Check for choices and remove them
    doc_to_text = styles_01(doc)
    return "Question: " + doc_to_text + "\nAnswer:"


def doc_to_choice(doc):
    return parse_choices(doc)


def doc_to_target(doc):
    target = doc["target"]
    try:
        if target in ["Yes", "No"]:
            return yes_no.index(target)
        else:
            return string.ascii_uppercase.index(target[1:-1])
        # else:
        #     return parse_choices(doc).index(target)

    except Exception as err:
        print("Full Doc")
        print(doc)
        print("Choices")
        print(parse_choices(doc))
        print("Error")
        print(err)
        import sys

        sys.exit()
