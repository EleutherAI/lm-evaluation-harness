import re

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


def get_choice_labels(choices):
    n_choices = len(choices)
    if n_choices < 26:  # A, B, C, ...
        return [chr(65 + i) for i in range(n_choices)]
    n_digits = len(str(n_choices))
    return [str(i + 1).zfill(n_digits) for i in range(n_choices)]


def choices_to_text(choices, choice_labels):
    return "\n".join(
        [
            f"{label.strip()}. {text.strip()}"
            for label, text in zip(choice_labels, choices)
        ]
    )


def get_choices_text_answer(choices, answer):
    if len(choices) == 0:
        return "", [" " + str(a).strip() for a in answer], None
    choice_labels = get_choice_labels(choices)
    choices_text = choices_to_text(choices, choice_labels)
    choice_labels = [" " + label for label in choice_labels]
    target = [choice_labels[i] for i in answer]
    return choices_text, target, choice_labels


def get_question_target(choices, answer, question, instruction="", directqa=True):
    choices_text, target, choice_labels = get_choices_text_answer(choices, answer)
    if directqa:
        question = (
            f"{instruction}\n\nQuestion: {question.strip()}\n{choices_text}\nAnswer:"
        )
    else:
        question = f"Question: {question.strip()}\n{choices_text}\n\n{instruction}"
    return question.strip(), target, choice_labels


def construct_prompt(instruction, opinion, question):
    return f"{instruction}\n\n{opinion}\n\n{question}"


def process_targets(targets):
    target = str(targets[0]).strip()
    if target.isdigit():
        target = str(int(target))
    return target


def process_docs(dataset, question_kwargs):
    def _helper(doc):
        question, targets, _ = get_question_target(
            doc["choices"], doc["answer"], doc["question"], **question_kwargs
        )
        return {
            **doc,
            "text": construct_prompt(doc["instruction"], doc["opinion"], question),
            "target": process_targets(targets),
        }

    return dataset.map(_helper)


def process_docs_direct(dataset):
    return process_docs(
        dataset,
        {
            "instruction": "Respond with a single uppercase letter (A-Z) or a numerical value (e.g., 9, 121). Provide no explanation or additional text.",
            "directqa": True,
        },
    )


def process_docs_cot(dataset):
    return process_docs(
        dataset,
        {
            "instruction": 'Think step by step. At the end, respond with "The final answer is [final_answer]", where [final_answer] is either a single uppercase letter (A-Z) or a numerical value (e.g., 9, 121).',
            "directqa": False,
        },
    )


@register_filter("caselaw-default-filter")
class DefaultCaselawFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_inst(inst, target):
            if target.isdigit():
                matches = list(re.finditer(r"\d+", inst))
                inst = str(int(matches[-1].group())) if matches else "[no match]"
            elif target.isupper():
                matches = list(re.finditer(r"[A-Z](?:[\s\.,*!?]|$)", inst))
                inst = (
                    matches[-1].group().strip().rstrip(".,!?:;*")
                    if matches
                    else "[no match]"
                )
            else:
                raise ValueError(
                    f"Target is neither digit nor uppercase letter: {target}"
                )
            return inst

        def filter_set(instances, doc):
            assert len(instances) == 1, (
                f"Expected 1 output response per example, got {len(instances)}"
            )
            return filter_inst(instances[0], doc["target"])

        return [filter_set(resp, doc) for resp, doc in zip(resps, docs)]
