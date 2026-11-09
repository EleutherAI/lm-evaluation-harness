from functools import partial

from datasets import Dataset


def process_docs(dataset, set_answer_type="bool"):
    FEATURES = ["title", "abstract", "question", "answer", "answer_type"]

    def _categorise_answer(answer_blob):
        if answer_blob["unanswerable"]:
            answer = "unanswerable"
            answer_type = "unanswerable"
            return answer, answer_type
        elif answer_blob["yes_no"]:
            answer = "yes"
            answer_type = "bool"
            return answer, answer_type
        elif answer_blob["free_form_answer"]:
            answer = answer_blob["free_form_answer"]
            answer_type = "free form answer"
            return answer, answer_type
        elif answer_blob["extractive_spans"]:
            answer = answer_blob["extractive_spans"]
            answer_type = "extractive_spans"
            return answer, answer_type
        elif answer_blob["yes_no"] is False:
            answer = "no"
            answer_type = "bool"
            return answer, answer_type

    def _flatten(doc):
        """Given a `doc`, flatten it out so that each JSON blob
        contains exactly one question and one answer. Logic taken from
        the reference implementation available at
        https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py
        """
        obs_list = {
            "title": [],
            "abstract": [],
            "question": [],
            "answer": [],
            "answer_type": [],
        }
        title = doc.pop("title")
        abstract = doc.pop("abstract")
        for question, answer_list in zip(doc["qas"]["question"], doc["qas"]["answers"]):
            for answer_blob in answer_list["answer"]:
                answer, answer_type = _categorise_answer(answer_blob)
                if answer_type == set_answer_type:
                    obs_list["title"].append(title)
                    obs_list["abstract"].append(abstract)
                    obs_list["question"].append(question)
                    obs_list["answer_type"].append(answer_type)
                    if isinstance(answer, list):
                        answer = ", ".join(answer)
                    obs_list["answer"].append(answer)

        return obs_list

    dataset = dataset.map(
        _flatten,
        remove_columns=[key for key in dataset.features.keys() if key not in FEATURES],
    )
    new_dataset = {}
    for key in dataset.features.keys():
        new_dataset[key] = [x for row in dataset[key] for x in row]

    return Dataset.from_dict(new_dataset)


process_docs_bool = partial(process_docs, set_answer_type="bool")
process_docs_freeform = partial(process_docs, set_answer_type="free form answer")
