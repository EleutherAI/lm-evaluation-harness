from lm_eval.utils import weighted_f1_score


def doc_to_choice(doc):
    choices = eval(doc["choices"])
    return choices


def doc_to_text(doc):
    output = """As an expert in {subject}, choose the most accurate answer to the question below.
Your goal is to select the correct option 'A', 'B', 'C', or 'D' by understanding the nuances of the topic.

Question: {question}
Choices:
        A: {choice1}
        B: {choice2}
        C: {choice3}
        D: {choice4}
Answer: """

    choices = eval(doc["choices"])
    text = output.format(
        subject=doc["subject"],
        question=doc["question"],
        choice1=choices[0],
        choice2=choices[1],
        choice3=choices[2],
        choice4=choices[3],
    )
    return text
