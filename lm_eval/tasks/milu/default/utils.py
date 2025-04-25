def doc_to_text(line):
    instruction = """## Task: You are a helpful and factual AI Assistant. The following is a Multiple Choice Question (MCQ) about {subject} ({domain}) in {language}. Now, choose the correct option.""".format(
        subject=line["subject"],
        domain=line["domain"],
        language=line["language"],
    ).strip()

    query = """
    {instruction}

    ## Question: {question}

    ## Choices:
    A. {option1}
    B. {option2}
    C. {option3}
    D. {option4}

    ## Answer:
    """.format(
        instruction=instruction,
        question=line["question"],
        option1=line["option1"],
        option2=line["option2"],
        option3=line["option3"],
        option4=line["option4"],
    ).strip()

    return query


def doc_to_target(line) -> int:
    return int(line["target"].replace('option', '')) - 1