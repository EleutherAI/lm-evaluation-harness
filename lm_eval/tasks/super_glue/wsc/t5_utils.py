import re
from typing import List


def doc_to_text(x):
    text = re.sub(r" X ", " *" + x["span2_text"] + "* ", _wsc_inputs(x))
    return "wsc: " + text


def _wsc_inputs(x):
    words = x["text"].split(" ")

    # We would need some special logic to handle the case where the pronoun is the
    # first or last word in the text. None of the examples in WSC seem to have
    # this, so we are ignoring these cases.
    assert x["span2_index"] > 0
    assert x["span2_index"] < len(words)
    pronoun_index = x["span2_index"]

    def create_input():
        assert words[pronoun_index] == x["span2_text"]

        return " ".join(
            [
                " ".join(words[:pronoun_index]),
                "X",
                " ".join(words[pronoun_index + 1 :]),
            ]
        )

    # Handle some special cases.
    if (
        x["text"]
        == 'The boy continued to whip the pony , and eventually the pony threw him over. John laughed out quite loud. "Good for him," he said. '
    ):
        return (
            "The boy continued to whip the pony , and eventually the pony threw "
            'him over. John laughed out quite loud. "Good for X ," he said.'
        )

    # Using the span2_index, we get 'use' instead of 'it'.
    if (
        x["text"]
        == "When they had eventually calmed down a bit , and had gotten home, Mr. Farley put the magic pebble in an iron safe . Some day they might want to use it , but really for now, what more could they wish for?"
    ):
        return (
            "When they had eventually calmed down a bit , and had gotten home, "
            "Mr. Farley put the magic pebble in an iron safe . Some day they might "
            "want to use X , but really for now, what more could they wish for?"
        )

    return create_input()


DETERMINERS = {
    "a",
    "an",
    "few",
    "her",
    "his",
    "each",
    "every",
    "many",
    "much",
    "my",
    "our",
    "some",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "which",
    "whose",
    "your",
}


def clean(s: str) -> str:
    """Ignore capitalization and determiners."""
    s = s.strip().lower()
    return " ".join([w for w in s.split(" ") if w not in DETERMINERS])


def process_results(docs: dict, resps: List):
    prediction = clean(resps[0])
    reference = clean(docs["span1_text"])

    if ("'" in prediction) != ("'" in reference):
        # referent is "Bob's hat" as predicting the referent.
        predicted_referent = False
    else:
        prediction_words = set(prediction.split(" "))
        referent_words = set(reference.split(" "))

        # Handle cases where the prediction is "fuzzy bunny" and the referent is
        # "bunny".
        predicted_referent = prediction_words.issubset(
            referent_words
        ) or referent_words.issubset(prediction_words)

    acc = 1.0 if predicted_referent == docs["label"] else 0.0
    return {"accuracy": acc}
