en_answer_mapping = {"true": "yes", "false": "no", True: "yes", False: "no"}


def process_docs(dataset):
    def remove_question_mark(text):
        text = text.strip()
        if text.endswith("?") or text.endswith("؟"):
            text = text[:-1]
        text = text.strip()

        return text

    def _helper(doc):
        doc["question"] = remove_question_mark(doc["question"])
        doc["target"] = en_answer_mapping[doc["answer"]]
        return doc

    return dataset.map(_helper)
