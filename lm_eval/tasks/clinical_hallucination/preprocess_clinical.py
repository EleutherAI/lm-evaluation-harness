# -*- coding: utf-8 -*-
"""Preprocessing functions for the clinical hallucination detection task.

Transforms MedQA-USMLE questions into a generative prompt format suitable
for hallucination analysis.  The model is asked to provide both an answer
and a clinical explanation so that the generated text contains enough
medical terminology for the hallucination detector to evaluate.
"""



def doc_to_text(doc) -> str:
    """Build a generative prompt from the MedQA document.

    The prompt asks the model to answer the clinical question AND provide
    a brief explanation, giving the hallucination detector enough text to
    analyze for fabricated medical terms.
    """
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join(f"{k}. {v}\n" for k, v in option_choices.items())
    return (
        f"Question: {doc['sent1']}\n"
        f"{answers}"
        f"Provide the correct answer letter and a brief clinical explanation "
        f"of the diagnosis and treatment rationale.\nAnswer:"
    )


def doc_to_target(doc) -> str:
    """Return the correct answer choice text as the reference.

    This serves as the ground-truth against which the model's generated
    explanation is compared for hallucination detection.
    """
    label = doc["label"]
    choices = ["ending0", "ending1", "ending2", "ending3"]
    answer_text = doc[choices[label]]
    # Include the answer letter and text as reference
    letter = chr(ord("A") + label)
    return f"{letter}. {answer_text}"
