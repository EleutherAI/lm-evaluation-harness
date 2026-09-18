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
    """Return the correct answer choice text, question text, and all option texts as reference.

    This provides a broader context against which the model's generated
    explanation is compared, ensuring that terms mentioned in the prompt
    or any options are not incorrectly flagged as hallucinations.
    """
    label = doc["label"]
    choices = ["ending0", "ending1", "ending2", "ending3"]
    answer_text = doc[choices[label]]
    letter = chr(ord("A") + label)

    # Extract the question (sent1) and optional sent2
    question = doc.get("sent1", "")
    if doc.get("sent2"):
        question += " " + doc["sent2"]

    options = " ".join(doc[c] for c in choices)

    return (
        f"Correct Answer: {letter}. {answer_text}\n"
        f"Question: {question}\n"
        f"Options: {options}"
    )
