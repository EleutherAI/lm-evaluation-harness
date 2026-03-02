"""Winogrande ``multiple_input`` preprocessing.

Winogrande is a fill-in-the-blank task: a sentence contains a "_" placeholder
and two candidate words. We split the sentence at the blank so each candidate
can be scored by how likely a single continuation is given the contexts.

Return contract (v0.5 ``multiple_input: true``):

    doc_to_text   -> list[str]  prefix + option for each candidate
    doc_to_choice -> list[str]  shared continuation (len 1, may be [""])
    doc_to_target -> int        0-based index of the correct candidate

Examples (options = ["vase", "table"], correct = 0):

    sentence        | doc_to_text                          | doc_to_choice | notes
    --------------- | ------------------------------------ | ------------- | -------------------------
    "The _ broke."  | ["The vase", "The table"]            | [" broke."]   | typical case
    "_ broke."      | ["vase", "table"]                    | [" broke."]   | empty prefix is valid
    "The big _"     | ["The big vase", "The big table"]    | [""]          | scores as unconditional LL
"""


def doc_to_target(doc) -> int:
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_choice(doc) -> list[str]:
    idx = doc["sentence"].index("_") + 1
    return [doc["sentence"][idx:].strip()]


def doc_to_text(doc) -> list[str]:
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]
