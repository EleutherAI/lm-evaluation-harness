import logging

from datasets import load_dataset


eval_logger = logging.getLogger(__name__)

# Define sequence lengths from 5 to 500 in increments of 5
DEFAULT_SEQ_LENGTHS = list(range(5, 505, 5))

# Group complexity classifications
TC0_GROUPS = {
    # Symmetric (solvable for n ≤ 4)
    "s3",
    "s4",
    # Alternating (solvable for n ≤ 4)
    "a3",
    "a4",
    # Cyclic (all are abelian, hence solvable)
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "c9",
    "c10",
    "c11",
    "c12",
    "c13",
    "c14",
    "c15",
    "c16",
    "c17",
    "c18",
    "c19",
    "c20",
    "c21",
    "c22",
    "c23",
    "c24",
    "c25",
    "c26",
    "c27",
    "c28",
    "c29",
    "c30",
    # Dihedral (all are solvable)
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
    "d8",
    "d9",
    "d10",
    "d11",
    "d12",
    "d13",
    "d14",
    "d15",
    "d16",
    "d17",
    "d18",
    "d19",
    "d20",
    # Quaternion (non-abelian 2-groups, solvable)
    "q8",
    "q16",
    "q32",
    # Frobenius (solvable)
    "f20",
    "f21",
    # Klein four-group (abelian)
    "v4",
    # Elementary abelian (direct products of cyclic groups)
    "z2_1",
    "z2_2",
    "z2_3",
    "z2_4",
    "z2_5",
    "z3_1",
    "z3_2",
    "z3_3",
    "z3_4",
    "z5_1",
    "z5_2",
    "z5_3",
    "z5_4",
    # Projective Special Linear (solvable cases)
    "psl2_2",
    "psl2_3",
}

NC1_GROUPS = {
    # Symmetric (non-solvable for n ≥ 5)
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    # Alternating (simple groups for n ≥ 5)
    "a5",
    "a6",
    "a7",
    "a8",
    "a9",
    # Projective Special Linear (simple groups)
    "psl2_4",
    "psl2_5",
    "psl2_7",
    "psl2_8",
    "psl2_9",
    "psl2_11",
    "psl3_2",
    "psl3_3",
    "psl3_4",
    "psl3_5",
    # Mathieu (sporadic simple groups)
    "m11",
    "m12",
}


def get_complexity_class(group_name: str) -> str:
    """Return the complexity class (TC0 or NC1) for a given group."""
    if group_name in TC0_GROUPS:
        return "TC0"
    elif group_name in NC1_GROUPS:
        return "NC1"
    else:
        return "Unknown"


def filter_by_sequence_length(dataset, min_length: int, max_length: int):
    """Filter dataset examples by sequence length."""
    return dataset.filter(lambda x: min_length <= x["sequence_length"] <= max_length)


def create_length_specific_dataset(
    group_name: str, target_length: int, split: str = "test"
):
    """Create a dataset filtered to a specific sequence length range."""
    # Load the dataset using name
    dataset = load_dataset(
        "BeeGass/Group-Theory-Collection", name=group_name, split=split
    )

    # Filter to target length with some tolerance
    # We use a window of ±2 to ensure we have enough samples
    filtered = filter_by_sequence_length(
        dataset, min_length=target_length - 2, max_length=target_length + 2
    )

    return filtered


def process_results(doc: dict, results) -> dict[str, float]:
    """Process model outputs and compute metrics for each sequence length."""
    # Initialize all metrics to -1 (indicating no data for that length)
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}

    # Get the actual sequence length from the document
    seq_len = doc["sequence_length"]

    # Find the closest evaluation length to bucket this result
    closest_length = min(DEFAULT_SEQ_LENGTHS, key=lambda x: abs(x - seq_len))

    # For loglikelihood tasks, results is a tuple (log_likelihood, is_greedy)
    # where is_greedy indicates if the target was the greedy choice
    if results and len(results) == 2:
        log_likelihood, is_greedy = results
        # Use accuracy (whether the greedy choice matches the target) as the metric
        score = float(is_greedy)
        metrics[str(closest_length)] = score
    elif results and len(results) == 1:
        # Sometimes results is wrapped in a list
        if isinstance(results[0], tuple) and len(results[0]) == 2:
            log_likelihood, is_greedy = results[0]
            score = float(is_greedy)
            metrics[str(closest_length)] = score

    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    """Aggregate metrics for a specific sequence length."""
    # Filter out -1 values (no data)
    valid_metrics = [x for x in metrics if x != -1]

    if not valid_metrics:
        # No samples for this length
        return -1

    # Return average accuracy
    return sum(valid_metrics) / len(valid_metrics)


# Generic dataset loader function
def load_group_dataset(group_name: str, **kwargs):
    """Load dataset for a specific group."""
    return load_dataset("BeeGass/Group-Theory-Collection", name=group_name)


# Custom dataset loader functions for each group
# We still need these as wrappers because the YAML files reference specific functions
# TC0 Groups (Solvable)


# Symmetric groups (TC0)
def s3_dataset(**kwargs):
    """Load S3 dataset (TC0 - solvable symmetric group)."""
    return load_group_dataset("s3", **kwargs)


def s4_dataset(**kwargs):
    """Load S4 dataset (TC0 - solvable symmetric group)."""
    return load_group_dataset("s4", **kwargs)


# Alternating groups (TC0)
def a3_dataset(**kwargs):
    """Load A3 dataset (TC0 - solvable alternating group)."""
    return load_group_dataset("a3", **kwargs)


def a4_dataset(**kwargs):
    """Load A4 dataset (TC0 - solvable alternating group)."""
    return load_group_dataset("a4", **kwargs)


# Cyclic groups (TC0 - all abelian)
def c2_dataset(**kwargs):
    """Load C2 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c2", **kwargs)


def c3_dataset(**kwargs):
    """Load C3 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c3", **kwargs)


def c4_dataset(**kwargs):
    """Load C4 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c4", **kwargs)


def c5_dataset(**kwargs):
    """Load C5 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c5", **kwargs)


def c6_dataset(**kwargs):
    """Load C6 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c6", **kwargs)


def c7_dataset(**kwargs):
    """Load C7 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c7", **kwargs)


def c8_dataset(**kwargs):
    """Load C8 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c8", **kwargs)


def c9_dataset(**kwargs):
    """Load C9 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c9", **kwargs)


def c10_dataset(**kwargs):
    """Load C10 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c10", **kwargs)


def c11_dataset(**kwargs):
    """Load C11 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c11", **kwargs)


def c12_dataset(**kwargs):
    """Load C12 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c12", **kwargs)


def c13_dataset(**kwargs):
    """Load C13 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c13", **kwargs)


def c14_dataset(**kwargs):
    """Load C14 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c14", **kwargs)


def c15_dataset(**kwargs):
    """Load C15 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c15", **kwargs)


def c16_dataset(**kwargs):
    """Load C16 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c16", **kwargs)


def c17_dataset(**kwargs):
    """Load C17 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c17", **kwargs)


def c18_dataset(**kwargs):
    """Load C18 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c18", **kwargs)


def c19_dataset(**kwargs):
    """Load C19 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c19", **kwargs)


def c20_dataset(**kwargs):
    """Load C20 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c20", **kwargs)


def c21_dataset(**kwargs):
    """Load C21 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c21", **kwargs)


def c22_dataset(**kwargs):
    """Load C22 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c22", **kwargs)


def c23_dataset(**kwargs):
    """Load C23 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c23", **kwargs)


def c24_dataset(**kwargs):
    """Load C24 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c24", **kwargs)


def c25_dataset(**kwargs):
    """Load C25 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c25", **kwargs)


def c26_dataset(**kwargs):
    """Load C26 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c26", **kwargs)


def c27_dataset(**kwargs):
    """Load C27 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c27", **kwargs)


def c28_dataset(**kwargs):
    """Load C28 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c28", **kwargs)


def c29_dataset(**kwargs):
    """Load C29 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c29", **kwargs)


def c30_dataset(**kwargs):
    """Load C30 dataset (TC0 - cyclic group)."""
    return load_group_dataset("c30", **kwargs)


# Dihedral groups (TC0 - all solvable)
def d3_dataset(**kwargs):
    """Load D3 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d3", **kwargs)


def d4_dataset(**kwargs):
    """Load D4 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d4", **kwargs)


def d5_dataset(**kwargs):
    """Load D5 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d5", **kwargs)


def d6_dataset(**kwargs):
    """Load D6 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d6", **kwargs)


def d7_dataset(**kwargs):
    """Load D7 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d7", **kwargs)


def d8_dataset(**kwargs):
    """Load D8 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d8", **kwargs)


def d9_dataset(**kwargs):
    """Load D9 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d9", **kwargs)


def d10_dataset(**kwargs):
    """Load D10 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d10", **kwargs)


def d11_dataset(**kwargs):
    """Load D11 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d11", **kwargs)


def d12_dataset(**kwargs):
    """Load D12 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d12", **kwargs)


def d13_dataset(**kwargs):
    """Load D13 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d13", **kwargs)


def d14_dataset(**kwargs):
    """Load D14 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d14", **kwargs)


def d15_dataset(**kwargs):
    """Load D15 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d15", **kwargs)


def d16_dataset(**kwargs):
    """Load D16 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d16", **kwargs)


def d17_dataset(**kwargs):
    """Load D17 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d17", **kwargs)


def d18_dataset(**kwargs):
    """Load D18 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d18", **kwargs)


def d19_dataset(**kwargs):
    """Load D19 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d19", **kwargs)


def d20_dataset(**kwargs):
    """Load D20 dataset (TC0 - dihedral group)."""
    return load_group_dataset("d20", **kwargs)


# Quaternion groups (TC0 - non-abelian 2-groups)
def q8_dataset(**kwargs):
    """Load Q8 dataset (TC0 - quaternion group)."""
    return load_group_dataset("q8", **kwargs)


def q16_dataset(**kwargs):
    """Load Q16 dataset (TC0 - quaternion group)."""
    return load_group_dataset("q16", **kwargs)


def q32_dataset(**kwargs):
    """Load Q32 dataset (TC0 - quaternion group)."""
    return load_group_dataset("q32", **kwargs)


# Frobenius groups (TC0)
def f20_dataset(**kwargs):
    """Load F20 dataset (TC0 - Frobenius group)."""
    return load_group_dataset("f20", **kwargs)


def f21_dataset(**kwargs):
    """Load F21 dataset (TC0 - Frobenius group)."""
    return load_group_dataset("f21", **kwargs)


# Klein four-group (TC0 - abelian)
def v4_dataset(**kwargs):
    """Load V4 dataset (TC0 - Klein four-group, isomorphic to Z2²)."""
    return load_group_dataset("v4", **kwargs)


# Elementary abelian groups (TC0)
def z2_1_dataset(**kwargs):
    """Load Z21 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z2_1", **kwargs)


def z2_2_dataset(**kwargs):
    """Load Z22 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z2_2", **kwargs)


def z2_3_dataset(**kwargs):
    """Load Z23 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z2_3", **kwargs)


def z2_4_dataset(**kwargs):
    """Load Z24 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z2_4", **kwargs)


def z2_5_dataset(**kwargs):
    """Load Z25 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z2_5", **kwargs)


def z3_1_dataset(**kwargs):
    """Load Z31 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z3_1", **kwargs)


def z3_2_dataset(**kwargs):
    """Load Z32 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z3_2", **kwargs)


def z3_3_dataset(**kwargs):
    """Load Z33 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z3_3", **kwargs)


def z3_4_dataset(**kwargs):
    """Load Z34 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z3_4", **kwargs)


def z5_1_dataset(**kwargs):
    """Load Z51 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z5_1", **kwargs)


def z5_2_dataset(**kwargs):
    """Load Z52 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z5_2", **kwargs)


def z5_3_dataset(**kwargs):
    """Load Z53 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z5_3", **kwargs)


def z5_4_dataset(**kwargs):
    """Load Z54 dataset (TC0 - elementary abelian group)."""
    return load_group_dataset("z5_4", **kwargs)


# PSL groups (TC0 - solvable cases)
def psl2_2_dataset(**kwargs):
    """Load PSL2(2) dataset (TC0 - solvable PSL group)."""
    return load_group_dataset("psl2_2", **kwargs)


def psl2_3_dataset(**kwargs):
    """Load PSL2(3) dataset (TC0 - solvable PSL group)."""
    return load_group_dataset("psl2_3", **kwargs)


# NC1 Groups (Non-Solvable)


# Symmetric groups (NC1)
def s5_dataset(**kwargs):
    """Load S5 dataset (NC1 - non-solvable symmetric group)."""
    return load_group_dataset("s5", **kwargs)


def s6_dataset(**kwargs):
    """Load S6 dataset (NC1 - non-solvable symmetric group)."""
    return load_group_dataset("s6", **kwargs)


def s7_dataset(**kwargs):
    """Load S7 dataset (NC1 - non-solvable symmetric group)."""
    return load_group_dataset("s7", **kwargs)


def s8_dataset(**kwargs):
    """Load S8 dataset (NC1 - non-solvable symmetric group)."""
    return load_group_dataset("s8", **kwargs)


def s9_dataset(**kwargs):
    """Load S9 dataset (NC1 - non-solvable symmetric group)."""
    return load_group_dataset("s9", **kwargs)


# Alternating groups (NC1)
def a5_dataset(**kwargs):
    """Load A5 dataset (NC1 - simple alternating group)."""
    return load_group_dataset("a5", **kwargs)


def a6_dataset(**kwargs):
    """Load A6 dataset (NC1 - simple alternating group)."""
    return load_group_dataset("a6", **kwargs)


def a7_dataset(**kwargs):
    """Load A7 dataset (NC1 - simple alternating group)."""
    return load_group_dataset("a7", **kwargs)


def a8_dataset(**kwargs):
    """Load A8 dataset (NC1 - simple alternating group)."""
    return load_group_dataset("a8", **kwargs)


def a9_dataset(**kwargs):
    """Load A9 dataset (NC1 - simple alternating group)."""
    return load_group_dataset("a9", **kwargs)


# PSL groups (NC1 - simple groups)
def psl2_4_dataset(**kwargs):
    """Load PSL2(4) dataset (NC1 - simple group)."""
    return load_group_dataset("psl2_4", **kwargs)


def psl2_5_dataset(**kwargs):
    """Load PSL2(5) dataset (NC1 - simple group)."""
    return load_group_dataset("psl2_5", **kwargs)


def psl2_7_dataset(**kwargs):
    """Load PSL2(7) dataset (NC1 - simple group)."""
    return load_group_dataset("psl2_7", **kwargs)


def psl2_8_dataset(**kwargs):
    """Load PSL2(8) dataset (NC1 - simple group)."""
    return load_group_dataset("psl2_8", **kwargs)


def psl2_9_dataset(**kwargs):
    """Load PSL2(9) dataset (NC1 - simple group)."""
    return load_group_dataset("psl2_9", **kwargs)


def psl2_11_dataset(**kwargs):
    """Load PSL2(11) dataset (NC1 - simple group)."""
    return load_group_dataset("psl2_11", **kwargs)


def psl3_2_dataset(**kwargs):
    """Load PSL3(2) dataset (NC1 - simple group)."""
    return load_group_dataset("psl3_2", **kwargs)


def psl3_3_dataset(**kwargs):
    """Load PSL3(3) dataset (NC1 - simple group)."""
    return load_group_dataset("psl3_3", **kwargs)


def psl3_4_dataset(**kwargs):
    """Load PSL3(4) dataset (NC1 - simple group)."""
    return load_group_dataset("psl3_4", **kwargs)


def psl3_5_dataset(**kwargs):
    """Load PSL3(5) dataset (NC1 - simple group)."""
    return load_group_dataset("psl3_5", **kwargs)


# Mathieu groups (NC1 - sporadic simple groups)
def m11_dataset(**kwargs):
    """Load M11 dataset (NC1 - Mathieu sporadic simple group)."""
    return load_group_dataset("m11", **kwargs)


def m12_dataset(**kwargs):
    """Load M12 dataset (NC1 - Mathieu sporadic simple group)."""
    return load_group_dataset("m12", **kwargs)
