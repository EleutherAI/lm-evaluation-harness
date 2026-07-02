"""
Utility functions for the Turkish Clinical Source Support evaluation task.

This task evaluates whether LLMs can identify evidence-based clinical
recommendations vs. unsafe/incorrect medical advice in Turkish.
"""

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process and validate the dataset documents.
    """
    def _helper(doc):
        # Validate that answer index is within range
        num_choices = sum(1 for key in doc.keys() if key.startswith("choice_"))
        assert 0 <= doc["answer"] < num_choices, (
            f"Answer index {doc['answer']} out of range (0-{num_choices - 1})"
        )
        return doc

    return dataset.map(_helper)


def doc_to_text(doc) -> str:
    """
    Format the question with answer choices as a multiple-choice prompt.
    """
    choices = [
        doc["choice_0"],
        doc["choice_1"],
        doc["choice_2"],
        doc["choice_3"],
    ]
    labels = ["A", "B", "C", "D"]

    prompt = (
        "Aşağıdaki klinik soruya kanıta dayalı en doğru yanıtı seçin.\n\n"
        f"Soru: {doc['question']}\n\n"
        "Seçenekler:\n"
    )
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nDoğru yanıt:"

    return prompt


def doc_to_choice(doc) -> list[str]:
    """
    Return answer choices as a list.
    """
    return [
        doc["choice_0"],
        doc["choice_1"],
        doc["choice_2"],
        doc["choice_3"],
    ]


def doc_to_target(doc) -> int:
    """
    Return the index of the correct answer.
    """
    return int(doc["answer"])
