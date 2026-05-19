import re
from typing import List, Tuple

import datasets


def load_dataset(**kwargs):
    """
    Load the graphwalks dataset with specific data file.

    Args:
        kwargs: Must contain 'data_file' key specifying which parquet file to load

    Returns:
        Dictionary with 'train' split containing the dataset
    """
    data_file = kwargs.get("data_file")
    if not data_file:
        raise ValueError("data_file must be specified in dataset_kwargs")

    dataset = datasets.load_dataset(
        "openai/graphwalks", data_files=data_file, split="train"
    )
    return {"train": dataset}


def extract_answer_list(response: str) -> Tuple[List[str], bool]:
    """
    Extract the answer list from a model response.

    Args:
        response: The model's generated response

    Returns:
        Tuple of (list of nodes, is_error)
        - list of nodes: extracted node IDs
        - is_error: True if parsing failed, False otherwise
    """
    # Get the very last line of the response (strip trailing newlines first)
    line = response.rstrip("\n").split("\n")[-1]

    # Check if formatted correctly
    if "Final Answer:" not in line:
        return [], True

    # Extract the list part using regex with capturing group
    match = re.search(r"Final Answer:\s*\[(.*)\]", line)
    if match:
        # Extract content between brackets using group(1)
        bracket_content = match.group(1)
        # Handle empty list case
        if not bracket_content.strip():
            return [], False
        # Split by comma and clean up whitespace and quotes
        result_list = [
            item.strip().strip("'\"")
            for item in bracket_content.split(",")
            if item.strip()
        ]
        return result_list, False
    else:
        return [], True


def extract_answer_list_flexible(response: str) -> Tuple[List[str], bool]:
    """
    Extract the answer list from a model response (flexible version).
    Searches backwards through all lines to find "Final Answer:" pattern.
    More lenient than extract_answer_list which only checks the last line.

    Args:
        response: The model's generated response

    Returns:
        Tuple of (list of nodes, is_error)
        - list of nodes: extracted node IDs
        - is_error: True if parsing failed, False otherwise
    """
    lines = response.rstrip("\n").split("\n")
    for line in reversed(lines):
        match = re.search(r"Final Answer:\s*\[(.*)\]", line)
        if match:
            # Extract content between brackets using group(1)
            bracket_content = match.group(1)
            # Handle empty list case
            if not bracket_content.strip():
                return [], False
            # Split by comma and clean up whitespace and quotes
            result_list = [
                item.strip().strip("'\"")
                for item in bracket_content.split(",")
                if item.strip()
            ]
            return result_list, False

    # No "Final Answer:" found anywhere
    return [], True


def process_results(doc, results):
    """
    Process results and compute set-based F1 scores.
    Returns both strict F1 (last line only) and flexible F1 (search all lines).

    Args:
        doc: Document containing ground truth answer_nodes
        results: List containing model generation

    Returns:
        Dictionary with f1 and flexible_f1 scores
    """
    # Extract model response (first element of results)
    response = results[0]

    # Get ground truth nodes
    gold_nodes = doc["answer_nodes"]

    # Parse the response using strict extraction
    predicted_nodes_strict, _ = extract_answer_list(response)
    sampled_set_strict = set(predicted_nodes_strict)
    truth_set = set(gold_nodes)

    # Calculate strict F1
    n_overlap_strict = len(sampled_set_strict & truth_set)
    n_sampled_strict = len(sampled_set_strict)
    n_golden = len(truth_set)

    recall_strict = n_overlap_strict / n_golden if n_golden > 0 else 0.0
    precision_strict = (
        n_overlap_strict / n_sampled_strict if n_sampled_strict > 0 else 0.0
    )
    f1_strict = (
        2 * (recall_strict * precision_strict) / (recall_strict + precision_strict)
        if (recall_strict + precision_strict) > 0
        else 0.0
    )

    # Parse the response using flexible extraction
    predicted_nodes_flexible, _ = extract_answer_list_flexible(response)
    sampled_set_flexible = set(predicted_nodes_flexible)

    # Calculate flexible F1
    n_overlap_flexible = len(sampled_set_flexible & truth_set)
    n_sampled_flexible = len(sampled_set_flexible)

    recall_flexible = n_overlap_flexible / n_golden if n_golden > 0 else 0.0
    precision_flexible = (
        n_overlap_flexible / n_sampled_flexible if n_sampled_flexible > 0 else 0.0
    )
    f1_flexible = (
        2
        * (recall_flexible * precision_flexible)
        / (recall_flexible + precision_flexible)
        if (recall_flexible + precision_flexible) > 0
        else 0.0
    )

    return {
        "f1": f1_strict,
        "flexible_f1": f1_flexible,
    }
