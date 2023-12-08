import string
from functools import partial

def doc_to_text_base(alphabet, style, doc):

    choices = doc["choices"]["text"]
    num = len(choices)

    letter_list = [style.format(letter) for letter in alphabet[0:num]]

    if "\t" in style:
        choice_string = "{}{}"
    else:
        choice_string = "{} {}"

    doc_to_text = "\n\n".join([
        "Question: "+doc["question"]+"\nAnswer:",
        ] + [
            choice_string.format(i,j) for i,j in zip(letter_list, choices)
        ]
    )

    return doc_to_text

# Full continuation
def choice_A(doc):
    return doc["choices"]["text"]

# Letters only
def choice_B(alphabet, style, doc):

    choices = doc["choices"]["text"]
    num = len(choices)

    letter_list = [style.format(letter) for letter in alphabet[0:num]]
    if "\t" in style:
        letter_list = [letter.replace("\t","") for letter in letter_list]

    return letter_list

# Letters + Full continuation
def choice_C(alphabet, style, doc):

    choices = doc["choices"]["text"]
    num = len(choices)

    letter_list = [style.format(letter) for letter in alphabet[0:num]]
    if "\t" not in style:
        letter_list = [letter+" " for letter in letter_list]

    return [letter+choice for letter, choice in zip(letter_list, choices)]

template_01 = partial(doc_to_text_base, string.ascii_lowercase, "({})")
choice_01a = choice_A
choice_01b = partial(choice_B, string.ascii_lowercase, "({})")
choice_01c = partial(choice_C, string.ascii_lowercase, "({})")
template_02 = partial(doc_to_text_base, string.ascii_lowercase, "{})")
choice_02a = choice_A
choice_02b = partial(choice_B, string.ascii_lowercase, "{})")
choice_02c = partial(choice_C, string.ascii_lowercase, "{})")
template_03 = partial(doc_to_text_base, string.ascii_lowercase, "{}.")
choice_03a = choice_A
choice_03b = partial(choice_B, string.ascii_lowercase, "{}.")
choice_03c = partial(choice_C, string.ascii_lowercase, "{}.")
template_04 = partial(doc_to_text_base, string.ascii_lowercase, "{}\t")
choice_04a = choice_A
choice_04b = partial(choice_B, string.ascii_lowercase, "{}\t")
choice_04c = partial(choice_C, string.ascii_lowercase, "{}\t")
template_05 = partial(doc_to_text_base, string.ascii_uppercase, "({})")
choice_05a = choice_A
choice_05b = partial(choice_B, string.ascii_uppercase, "({})")
choice_05c = partial(choice_C, string.ascii_uppercase, "({})")
template_06 = partial(doc_to_text_base, string.ascii_uppercase, "{})")
choice_06a = choice_A
choice_06b = partial(choice_B, string.ascii_uppercase, "{})")
choice_06c = partial(choice_C, string.ascii_uppercase, "{})")
template_07 = partial(doc_to_text_base, string.ascii_uppercase, "{}.")
choice_07a = choice_A
choice_07b = partial(choice_B, string.ascii_uppercase, "{}.")
choice_07c = partial(choice_C, string.ascii_uppercase, "{}.")
template_08 = partial(doc_to_text_base, string.ascii_uppercase, "{}\t")
choice_08a = choice_A
choice_08b = partial(choice_B, string.ascii_uppercase, "{}\t")
choice_08c = partial(choice_C, string.ascii_uppercase, "{}\t")


