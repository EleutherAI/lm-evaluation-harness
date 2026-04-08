import ast


def doc_to_choice(doc):
    choices = ast.literal_eval(doc["choices"])
    return choices


def doc_to_text(doc):
    output = """You are a highly knowledgeable and intelligent artificial intelligence
                model answers multiple-choice questions about '{subject}'

                Question: '''{question}'''

                Choices:
                        A: ''{choice1}'''
                        B: ''{choice2}'''
                        C: ''{choice3}'''
                        D: ''{choice4}'''

                Answer:  """

    choices = ast.literal_eval(doc["choices"])
    text = output.format(
        subject=doc["subject"],
        question=doc["question"],
        choice1=choices[0],
        choice2=choices[1],
        choice3=choices[2],
        choice4=choices[3],
    )
    return text
