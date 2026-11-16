import hashlib
import random
import re

import datasets


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def process_arc(dataset: datasets.Dataset) -> datasets.Dataset:
    def _subprocess(doc):
        long_prompt = ""
        for shot in range(1, 26):
            question = doc[f"arc_question_shot_{shot}"]
            doc.pop(f"arc_question_shot_{shot}")
            answer_lab = doc[f"arc_answerKey_shot_{shot}"]
            doc.pop(f"arc_answerKey_shot_{shot}")
            answer_idx = doc[f"arc_choices_shot_{shot}"]["label"].index(answer_lab)
            answer = doc[f"arc_choices_shot_{shot}"]["text"][answer_idx]
            doc.pop(f"arc_choices_shot_{shot}")
            doc.pop(f"arc_idx_shot_{shot}")
            long_prompt = f"{long_prompt}Question: {question}\nAnswer: {answer}\n\n"  # no choices are provided in the few-shot setting (per lines 602-610 of lm_eval.api.task)
        doc["twentyfive_shot_preprompt"] = long_prompt
        doc.pop("alltwentyfiveshot_longprompt")
        doc["original_hash"] = hash_string(doc["question"])

        # permute choices randomly without replacement (the new answer label will never be the answer label recorded in the original benchmarks)
        original_answer_idx = doc["choices"]["label"].index(doc["answerKey"])
        correct_answer_text = doc["choices"]["text"][original_answer_idx]
        new_answer_idx = original_answer_idx

        while new_answer_idx is original_answer_idx:
            random.shuffle(doc["choices"]["text"])
            new_answer_idx = doc["choices"]["text"].index(correct_answer_text)
        doc["answerKey"] = doc["choices"]["label"][new_answer_idx]

        return doc

    return dataset.map(_subprocess)


def process_hellaswag(dataset: datasets.Dataset) -> datasets.Dataset:
    def process_txt(text):  # mirrored from hellaswag task
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def _preprocess(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        doc.pop("ctx_a")
        doc.pop("ctx_b")
        doc.pop("ctx")
        doc["query"] = process_txt(doc["activity_label"] + ": " + ctx)

        # permute choices randomly without replacement (the new answer label will never be the answer label recorded in the original benchmarks)
        original_answer_idx = int(doc["label"])
        correct_answer_text = doc["endings"][original_answer_idx]
        new_answer_idx = original_answer_idx
        while new_answer_idx is original_answer_idx:
            random.shuffle(doc["endings"])
            new_answer_idx = doc["endings"].index(correct_answer_text)
        doc["label"] = str(new_answer_idx)

        doc["choices"] = [process_txt(ending) for ending in doc["endings"]]
        doc["gold"] = int(doc["label"])
        doc.pop("activity_label")
        doc.pop("endings")

        long_prompt = ""
        for shot in range(1, 11):
            ctx = (
                doc[f"hellaswag_ctx_a_shot_{shot}"]
                + " "
                + doc[f"hellaswag_ctx_b_shot_{shot}"].capitalize()
            )
            doc.pop(f"hellaswag_ctx_a_shot_{shot}")
            doc.pop(f"hellaswag_ctx_b_shot_{shot}")
            doc.pop(f"hellaswag_ctx_shot_{shot}")
            question = process_txt(
                doc[f"hellaswag_activity_labels_shot_{shot}"] + ": " + ctx
            )
            ending = process_txt(
                doc[f"hellaswag_endings_shot_{shot}"][
                    int(doc[f"hellaswag_label_shot_{shot}"])
                ]
            )
            doc.pop(f"hellaswag_activity_labels_shot_{shot}")
            doc.pop(f"hellaswag_endings_shot_{shot}")
            doc.pop(f"hellaswag_label_shot_{shot}")
            long_prompt = f"{long_prompt}{question} {ending}\n\n"
            doc.pop(f"hellaswag_ind_shot_{shot}")
            doc.pop(f"hellaswag_source_id_shot_{shot}")
            doc.pop(f"hellaswag_split_shot_{shot}")
            doc.pop(f"hellaswag_split_type_shot_{shot}")

        doc["original_hash"] = hash_string(doc["query"])
        doc["ten_shot_preprompt"] = long_prompt
        doc.pop("alltenshot_longprompt")
        return doc

    return dataset.map(_preprocess)


def process_mmlu(dataset: datasets.Dataset) -> datasets.Dataset:
    def _subprocess(doc):
        choices = ["A", "B", "C", "D"]
        long_prompt = f"The following are multiple choice questions (with answers) about {' '.join(doc['subject'].split('_'))}.\n\n"
        for shot in range(1, 6):
            question = doc[f"mmlu_question_shot_{shot}"].strip()
            doc.pop(f"mmlu_question_shot_{shot}")
            answer = choices[int(doc[f"mmlu_answers_shot_{shot}"])]
            choice_A = doc[f"mmlu_choices_shot_{shot}"][0]
            choice_B = doc[f"mmlu_choices_shot_{shot}"][1]
            choice_C = doc[f"mmlu_choices_shot_{shot}"][2]
            choice_D = doc[f"mmlu_choices_shot_{shot}"][3]

            doc.pop(f"mmlu_choices_shot_{shot}")
            doc.pop(f"mmlu_answers_shot_{shot}")
            doc.pop(f"mmlu_ind_shot_{shot}")

            long_prompt = f"{long_prompt}{question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\nAnswer: {answer}\n\n"  # choices are provided in the mmlu few-shot regime, unlike other benchmarks.

        doc["original_hash"] = hash_string(doc["question"])
        doc["five_shot_preprompt"] = long_prompt
        doc.pop("allfiveshot_longprompt")

        # permute choices randomly without replacement (the new answer label will never be the answer label recorded in the original benchmarks)
        original_answer_idx = int(doc["answer"])
        correct_answer_text = doc["choices"][original_answer_idx]
        new_answer_idx = original_answer_idx

        while new_answer_idx is original_answer_idx:
            random.shuffle(doc["choices"])
            new_answer_idx = doc["choices"].index(correct_answer_text)
        doc["answer"] = new_answer_idx

        return doc

    return dataset.map(_subprocess)


def process_truthfulqa(dataset: datasets.Dataset) -> datasets.Dataset:
    def _subprocess(
        doc,
    ):  # currently only permuting the mc1 targets as metabench does not use mc2 targets.
        original_answer_idx = 0  # always 0 in truthfulqa
        correct_answer_text = doc["mc1_targets"]["choices"][original_answer_idx]
        new_answer_idx = original_answer_idx

        while new_answer_idx is original_answer_idx:
            random.shuffle(doc["mc1_targets"]["choices"])
            new_answer_idx = doc["mc1_targets"]["choices"].index(correct_answer_text)

        labels = [0] * len(doc["mc1_targets"]["labels"])
        labels[new_answer_idx] = 1
        doc["original_hash"] = hash_string(doc["question"])
        doc["mc1_targets"]["labels"] = labels
        doc["answer"] = new_answer_idx

        return doc

    return dataset.map(_subprocess)


def process_winogrande(dataset: datasets.Dataset) -> datasets.Dataset:
    def _subprocess(doc):
        long_prompt = ""
        for shot in range(1, 6):
            if doc[f"winogrande_answer_shot_{shot}"] == "1":
                answer = doc[f"winogrande_option1_shot_{shot}"]
            elif doc[f"winogrande_answer_shot_{shot}"] == "2":
                answer = doc[f"winogrande_option2_shot_{shot}"]
            else:
                raise ValueError("Answer not recognised.")

            question = doc[f"winogrande_prompt_shot_{shot}"].replace("_", answer)

            doc.pop(f"winogrande_prompt_shot_{shot}")
            doc.pop(f"winogrande_answer_shot_{shot}")
            doc.pop(f"winogrande_idx_shot_{shot}")
            doc.pop(f"winogrande_option1_shot_{shot}")
            doc.pop(f"winogrande_option2_shot_{shot}")

            long_prompt = f"{long_prompt}{question}\n\n"
        sentence = doc["sentence"]
        doc["original_hash"] = hash_string(doc["sentence"])
        doc["sentence"] = f"{long_prompt}{sentence}"
        doc.pop("allfiveshot_longprompt")

        # permute choices by swapping them
        option1 = doc["option1"]
        option2 = doc["option2"]
        answer = doc["answer"]

        doc["option1"] = option2
        doc["option2"] = option1

        if answer == "1":
            doc["answer"] = "2"
        elif answer == "2":
            doc["answer"] = "1"

        return doc

    return dataset.map(_subprocess)


def winogrande_doc_to_text(doc):  # Mirrored from the winogrande task
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def winogrande_doc_to_target(doc):  # Mirrored from the winogrande task
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def winogrande_doc_to_choice(doc):  # Mirrored from the winogrande task
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]
