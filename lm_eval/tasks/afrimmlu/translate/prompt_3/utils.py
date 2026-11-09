import ast

# referenced as `!function utils.weighted_f1_score` in the task configs;
# keep the import even though it looks unused (noqa guards against ruff --fix)
from lm_eval.utils import weighted_f1_score  # noqa: F401


def doc_to_choice(doc):
    choices = ast.literal_eval(doc["choices"])
    return choices


def doc_to_text(doc):
    output = """You are a subject matter expert in {subject}.

  Utilizing your expertise in {subject}, answer the following multiple-choice question
  by picking ''A'', ''B'', ''C'', or ''D''.

Question: {question}
Choices:
        A: {choice1}
        B: {choice2}
        C: {choice3}
        D: {choice4}
Answer: """

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
