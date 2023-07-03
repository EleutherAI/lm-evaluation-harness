def doc_to_text(doc):
    return (
        "Sentence 1: {}\nSentence 2: {}\nQuestion: Is the word '{}' used in the same way in the"
        " two sentences above?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
            doc["sentence1"][doc["start1"] : doc["end1"]],
        )
    )


def doc_to_target(doc):
    return " {}".format({0: "no", 1: "yes"}[doc["label"]])
