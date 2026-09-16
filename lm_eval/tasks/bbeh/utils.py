"""Answer extraction, scoring and subtask filtering for BBEH (BIG-Bench Extra Hard).

The extraction and fuzzy-matching logic is adapted from the official BBEH
evaluation script so that scores line up with the reference:
https://github.com/google-deepmind/bbeh/blob/main/bbeh/evaluate.py
"""

import contextlib
from functools import partial


def strip_latex(response: str) -> str:
    """Strip common LaTeX wrappers (dollar signs, ``boxed{}``, ``text{}``) from an answer."""
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    """Extract the final answer that follows an ``The answer is`` prefix."""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    answer = answer.removesuffix(".")
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """BBEH fuzzy match: handles ``(a)``/``a``, numbers, quotes and brackets."""
    if prediction == reference:
        return True

    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers
    with contextlib.suppress(ValueError):
        if float(prediction) == float(reference):
            return True

    # quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    return prediction.endswith("?") and prediction[:-1] == reference


def preprocess_sample(sample: str) -> str:
    """Normalise a model response down to the comparable answer string."""
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    """Normalise the gold target the same way as the prediction."""
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    """Return whether a model response matches the BBEH reference answer."""
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """lm-eval hook: score a single generated response against the BBEH target."""
    response = results[0]
    correct = evaluate_correctness(response, doc["target"])
    return {"exact_match": float(correct)}


def process_docs(dataset, task_name):
    """Filter the flat BBEH dataset down to the rows of a single subtask."""
    return dataset.filter(lambda example: example["task"] == task_name)


def filter_mini(dataset):
    """Keep only the official BBEH-mini subset (the ``mini`` flag is set)."""
    return dataset.filter(lambda example: str(example["mini"]) == "1")


# Per-subtask doc filters (the dataset's `task` column uses spaces, e.g. "boardgame qa").
process_boardgame_qa = partial(process_docs, task_name="boardgame qa")
process_boolean_expressions = partial(process_docs, task_name="boolean expressions")
process_buggy_tables = partial(process_docs, task_name="buggy tables")
process_causal_understanding = partial(process_docs, task_name="causal understanding")
process_disambiguation_qa = partial(process_docs, task_name="disambiguation qa")
process_dyck_languages = partial(process_docs, task_name="dyck languages")
process_geometric_shapes = partial(process_docs, task_name="geometric shapes")
process_hyperbaton = partial(process_docs, task_name="hyperbaton")
process_linguini = partial(process_docs, task_name="linguini")
process_movie_recommendation = partial(process_docs, task_name="movie recommendation")
process_multistep_arithmetic = partial(process_docs, task_name="multistep arithmetic")
process_nycc = partial(process_docs, task_name="nycc")
process_object_counting = partial(process_docs, task_name="object counting")
process_object_properties = partial(process_docs, task_name="object properties")
process_sarc_triples = partial(process_docs, task_name="sarc triples")
process_shuffled_objects = partial(process_docs, task_name="shuffled objects")
process_spatial_reasoning = partial(process_docs, task_name="spatial reasoning")
process_sportqa = partial(process_docs, task_name="sportqa")
process_temporal_sequence = partial(process_docs, task_name="temporal sequence")
process_time_arithmetic = partial(process_docs, task_name="time arithmetic")
process_web_of_lies = partial(process_docs, task_name="web of lies")
process_word_sorting = partial(process_docs, task_name="word sorting")
process_zebra_puzzles = partial(process_docs, task_name="zebra puzzles")
