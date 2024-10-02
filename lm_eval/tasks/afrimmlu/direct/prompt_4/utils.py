from lm_eval.utils import weighted_f1_score


def doc_to_choice(doc, lang):
    choices = eval(doc["choices"])
    return choices


def doc_to_text(doc, lang):
    output = """You are a subject matter expert in {subject} with professional working proficiency in {lang}

  Analyze each {lang} question critically and determine the most correct option based on your understanding of the s
  ubject matter.

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
        lang=lang,
        question=doc["question"],
        choice1=choices[0],
        choice2=choices[1],
        choice3=choices[2],
        choice4=choices[3],
    )
    return text
