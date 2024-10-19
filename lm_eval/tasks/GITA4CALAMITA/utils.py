def doc_to_text(x):

    PRE_PROMPT = "The story is as follows:"
    POST_PROMPT = "Is the story plausible?"

    instance = PRE_PROMPT + "\n"

    for sentence in x["sentences"]:
        instance += f'{sentence} '

    instance += "\n"

    instance += POST_PROMPT

    return instance