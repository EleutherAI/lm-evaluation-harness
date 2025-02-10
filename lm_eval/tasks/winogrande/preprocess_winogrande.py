def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]


def doc_to_text_gen(doc):
    sentence = doc["sentence"]
    is_noun = doc["option1"][0].isupper()
    question = "Who" if is_noun else "What"
    question = f"Question: {question}" + doc["sentence"].split("_")[-1]
    return (
        "Given the following question and two candidate answers (A and B), choose the best answer"
        + sentence
        + "\n"
        + question
        + "\n"
        + f"A. {doc['option1']}"
        + "\n"
        + f"B. {doc['option2']}"
        + 'Your response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A or B.'
    )
