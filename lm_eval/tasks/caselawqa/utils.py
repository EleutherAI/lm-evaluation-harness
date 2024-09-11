
def get_choice_labels(choices):
    n_choices = len(choices)
    if n_choices < 26: # A, B, C, ...
        return [chr(65 + i) for i in range(n_choices)]
    n_digits = len(str(n_choices))
    return [str(i+1).zfill(n_digits) for i in range(n_choices)]


def choices_to_text(choices, choice_labels):
    return '\n'.join([f"{label.strip()}. {text.strip()}" for label, text in zip(choice_labels, choices)])


def get_choices_text_answer(choices, answer):
    if len(choices) == 0:
        return '', [' ' + str(a).strip() for a in answer], None
    choice_labels = get_choice_labels(choices)
    choices_text = choices_to_text(choices, choice_labels)
    choice_labels = [' ' + label for label in choice_labels]
    target = [choice_labels[i] for i in answer]
    return choices_text, target, choice_labels


def get_question_target(choices, answer, question):
    choices_text, target, choice_labels = get_choices_text_answer(choices, answer)
    question = f"Question: {question.strip()}\n{choices_text}\nAnswer:"
    return question, target, choice_labels


def construct_prompt(instruction, opinion, question):
    return f"{instruction}\n\n{opinion}\n\n{question}"


def doc_to_text(doc):
    question, _, _ = get_question_target(doc['choices'], doc['answer'], doc['question'])
    return construct_prompt(doc['instruction'], doc['opinion'], question)


def doc_to_answer(doc):
    _, target, _ = get_question_target(doc['choices'], doc['answer'], doc['question'])
    return target[0]