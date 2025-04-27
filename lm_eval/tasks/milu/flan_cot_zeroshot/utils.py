def doc_to_text(line):
    return """
    ## Question: {question}

    ## Choices:
    A. {option1}
    B. {option2}
    C. {option3}
    D. {option4}
    """.format(
        question=line["question"],
        option1=line["option1"],
        option2=line["option2"],
        option3=line["option3"],
        option4=line["option4"],
    ).strip()


def doc_to_target(line) -> int:
    return int(line["target"].replace('option', '')) - 1