import sys

task = sys.argv[4]

def doc_to_text(x):

    if task == 'story_class':

        PRE_PROMPT = "The story is as follows:"
        POST_PROMPT = "Is the story plausible?"

        instance = PRE_PROMPT + "\n"
        for sentence in x["sentences"]:
            instance += f'{sentence} '

        instance += "\n"

        instance += POST_PROMPT

        return instance

    elif task == 'conflict_detec':

        PRE_PROMPT = "The story is as follows: "
        POST_PROMPT = "The conflicting sentence and the breakpoint are:"

        instance = PRE_PROMPT + "\n"

        for i, sentence in enumerate(x["sentences"]):
            instance += f'{i}. {sentence}\n'

        instance += "\n"

        instance += POST_PROMPT

        return instance
    
    elif task == 'physical_state':

        PRE_PROMPT = "The story is as follows: "
        POST_PROMPT = "The physical state that causes the conflict in the implausible story is: "

        instance = PRE_PROMPT + "\n"

        for sentence in x["sentences"]:
            instance += f'{sentence} '

        instance += "\n"

        instance += POST_PROMPT

        return instance


def doc_to_target(x):
    if 'confl_sents' in x and len(x['confl_sents']) > 0:
        return f"{x['confl_sents'][0]} and {x['breakpoint']}"
    else:
        return f"None"
#def doc_to_target(x):
#    return f"{x['confl_sents'][0]} and {x['breakpoint']}"