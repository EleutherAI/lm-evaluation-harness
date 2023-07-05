import re
from lm_eval.utils import general_detokenize


def partial_context(doc, option):
    # Substitute the pronoun in the sentence with the specified option
    # and ignore everything after.
    pronoun_loc = doc["sentence"].index("_")
    return doc["sentence"][:pronoun_loc] + option


def partial_target(doc):
    # The target is everything after the document specified pronoun.
    pronoun_loc = doc["sentence"].index("_") + 1
    return " " + doc["sentence"][pronoun_loc:].strip()
