#!/usr/bin/env python3
"""Generate all task YAML files for the permutation benchmark.

Run from anywhere; the files are written next to this script::

    python lm_eval/tasks/permutation_benchmark/generate_tasks.py

The output of this script is committed to the repository, so it must already
satisfy the repo pre-commit hooks (trailing newline, no trailing whitespace).
"""

from pathlib import Path


HERE = Path(__file__).resolve().parent

# Sequence lengths that each task reports a metric for.
SEQ_LENGTHS = range(5, 505, 5)

# Group information with descriptions
GROUP_INFO = {
    # TC0 Groups (Solvable)
    "s3": ("S3", "symmetric group on 3 elements", "tc0"),
    "s4": ("S4", "symmetric group on 4 elements", "tc0"),
    "a3": ("A3", "alternating group on 3 elements", "tc0"),
    "a4": ("A4", "alternating group on 4 elements", "tc0"),
    "c2": ("C2", "cyclic group of order 2", "tc0"),
    "c3": ("C3", "cyclic group of order 3", "tc0"),
    "c4": ("C4", "cyclic group of order 4", "tc0"),
    "c5": ("C5", "cyclic group of order 5", "tc0"),
    "c6": ("C6", "cyclic group of order 6", "tc0"),
    "c7": ("C7", "cyclic group of order 7", "tc0"),
    "c8": ("C8", "cyclic group of order 8", "tc0"),
    "c9": ("C9", "cyclic group of order 9", "tc0"),
    "c10": ("C10", "cyclic group of order 10", "tc0"),
    "c11": ("C11", "cyclic group of order 11", "tc0"),
    "c12": ("C12", "cyclic group of order 12", "tc0"),
    "c13": ("C13", "cyclic group of order 13", "tc0"),
    "c14": ("C14", "cyclic group of order 14", "tc0"),
    "c15": ("C15", "cyclic group of order 15", "tc0"),
    "c16": ("C16", "cyclic group of order 16", "tc0"),
    "c17": ("C17", "cyclic group of order 17", "tc0"),
    "c18": ("C18", "cyclic group of order 18", "tc0"),
    "c19": ("C19", "cyclic group of order 19", "tc0"),
    "c20": ("C20", "cyclic group of order 20", "tc0"),
    "c21": ("C21", "cyclic group of order 21", "tc0"),
    "c22": ("C22", "cyclic group of order 22", "tc0"),
    "c23": ("C23", "cyclic group of order 23", "tc0"),
    "c24": ("C24", "cyclic group of order 24", "tc0"),
    "c25": ("C25", "cyclic group of order 25", "tc0"),
    "c26": ("C26", "cyclic group of order 26", "tc0"),
    "c27": ("C27", "cyclic group of order 27", "tc0"),
    "c28": ("C28", "cyclic group of order 28", "tc0"),
    "c29": ("C29", "cyclic group of order 29", "tc0"),
    "c30": ("C30", "cyclic group of order 30", "tc0"),
    "d3": ("D3", "dihedral group of order 6", "tc0"),
    "d4": ("D4", "dihedral group of order 8", "tc0"),
    "d5": ("D5", "dihedral group of order 10", "tc0"),
    "d6": ("D6", "dihedral group of order 12", "tc0"),
    "d7": ("D7", "dihedral group of order 14", "tc0"),
    "d8": ("D8", "dihedral group of order 16", "tc0"),
    "d9": ("D9", "dihedral group of order 18", "tc0"),
    "d10": ("D10", "dihedral group of order 20", "tc0"),
    "d11": ("D11", "dihedral group of order 22", "tc0"),
    "d12": ("D12", "dihedral group of order 24", "tc0"),
    "d13": ("D13", "dihedral group of order 26", "tc0"),
    "d14": ("D14", "dihedral group of order 28", "tc0"),
    "d15": ("D15", "dihedral group of order 30", "tc0"),
    "d16": ("D16", "dihedral group of order 32", "tc0"),
    "d17": ("D17", "dihedral group of order 34", "tc0"),
    "d18": ("D18", "dihedral group of order 36", "tc0"),
    "d19": ("D19", "dihedral group of order 38", "tc0"),
    "d20": ("D20", "dihedral group of order 40", "tc0"),
    "q8": ("Q8", "quaternion group of order 8", "tc0"),
    "q16": ("Q16", "quaternion group of order 16", "tc0"),
    "q32": ("Q32", "quaternion group of order 32", "tc0"),
    "f20": ("F20", "Frobenius group of order 20", "tc0"),
    "f21": ("F21", "Frobenius group of order 21", "tc0"),
    "v4": ("V4", "Klein four-group", "tc0"),
    "z2_1": ("Z_2^1", "elementary abelian group", "tc0"),
    "z2_2": ("Z_2^2", "elementary abelian group", "tc0"),
    "z2_3": ("Z_2^3", "elementary abelian group", "tc0"),
    "z2_4": ("Z_2^4", "elementary abelian group", "tc0"),
    "z2_5": ("Z_2^5", "elementary abelian group", "tc0"),
    "z3_1": ("Z_3^1", "elementary abelian group", "tc0"),
    "z3_2": ("Z_3^2", "elementary abelian group", "tc0"),
    "z3_3": ("Z_3^3", "elementary abelian group", "tc0"),
    "z3_4": ("Z_3^4", "elementary abelian group", "tc0"),
    "z5_1": ("Z_5^1", "elementary abelian group", "tc0"),
    "z5_2": ("Z_5^2", "elementary abelian group", "tc0"),
    "z5_3": ("Z_5^3", "elementary abelian group", "tc0"),
    "z5_4": ("Z_5^4", "elementary abelian group", "tc0"),
    "psl2_2": ("PSL(2,2)", "projective special linear group", "tc0"),
    "psl2_3": ("PSL(2,3)", "projective special linear group", "tc0"),
    # NC1 Groups (Non-Solvable)
    "s5": ("S5", "symmetric group on 5 elements", "nc1"),
    "s6": ("S6", "symmetric group on 6 elements", "nc1"),
    "s7": ("S7", "symmetric group on 7 elements", "nc1"),
    "s8": ("S8", "symmetric group on 8 elements", "nc1"),
    "s9": ("S9", "symmetric group on 9 elements", "nc1"),
    "a5": ("A5", "alternating group on 5 elements", "nc1"),
    "a6": ("A6", "alternating group on 6 elements", "nc1"),
    "a7": ("A7", "alternating group on 7 elements", "nc1"),
    "a8": ("A8", "alternating group on 8 elements", "nc1"),
    "a9": ("A9", "alternating group on 9 elements", "nc1"),
    "psl2_4": ("PSL(2,4)", "projective special linear group", "nc1"),
    "psl2_5": ("PSL(2,5)", "projective special linear group", "nc1"),
    "psl2_7": ("PSL(2,7)", "projective special linear group", "nc1"),
    "psl2_8": ("PSL(2,8)", "projective special linear group", "nc1"),
    "psl2_9": ("PSL(2,9)", "projective special linear group", "nc1"),
    "psl2_11": ("PSL(2,11)", "projective special linear group", "nc1"),
    "psl3_2": ("PSL(3,2)", "projective special linear group", "nc1"),
    "psl3_3": ("PSL(3,3)", "projective special linear group", "nc1"),
    "psl3_4": ("PSL(3,4)", "projective special linear group", "nc1"),
    "psl3_5": ("PSL(3,5)", "projective special linear group", "nc1"),
    "m11": ("M11", "Mathieu group", "nc1"),
    "m12": ("M12", "Mathieu group", "nc1"),
}


def generate_metric_list() -> str:
    """Generate the per-sequence-length metric list for a task config."""
    return "\n".join(
        f'''  - metric: "{length}"
    aggregation: !function group_composition_utils.aggregate_metrics
    higher_is_better: true'''
        for length in SEQ_LENGTHS
    )


def generate_aggregate_metric_list() -> str:
    """Generate the per-sequence-length aggregate metric list for a group config."""
    return "\n".join(
        f'''  - metric: "{length}"
    weight_by_size: false'''
        for length in SEQ_LENGTHS
    )


def generate_task_yaml(
    group_id: str, group_name: str, group_desc: str, complexity: str
) -> str:
    """Generate YAML content for a single task."""
    # `doc_to_text` uses a `|-` block scalar so the rendered context does not end
    # in whitespace: for `loglikelihood` tasks the leading space of
    # `doc_to_target` is what separates the context from the continuation.
    return f"""tag:
  - group_theory
  - permutation_composition
  - {complexity}
task: {group_id}_composition
dataset_path: ""
dataset_name: ""
output_type: loglikelihood
test_split: test
custom_dataset: !function group_composition_utils.{group_id}_dataset
doc_to_text: |-
  You are given a sequence of permutations from the group {group_name} ({group_desc}), identified by their integer IDs. Your task is to compute their composed product.

  The composition must be performed sequentially from right to left, following the standard mathematical convention (p_n ∘ ... ∘ p_2 ∘ p_1).

  Sequence: {{{{input_sequence}}}}

  Question: What is the single integer ID of the final composed permutation? Your response must be only the integer.

  Answer:
doc_to_target: " {{{{target}}}}"
process_results: !function group_composition_utils.process_results
metric_list:
{generate_metric_list()}
metadata:
  version: 1.0
"""


def generate_group_yaml(name: str, group_ids: list[str]) -> str:
    """Generate YAML content for a complexity-class group file (TC0 or NC1)."""
    tasks = "\n".join(f"  - {group_id}_composition" for group_id in group_ids)
    return f"""group: {name}_groups
task:
{tasks}
aggregate_metric_list:
{generate_aggregate_metric_list()}
metadata:
  version: 1.0
  description: Permutation composition tasks for {name.upper()} complexity class
"""


def generate_benchmark_yaml() -> str:
    """Generate YAML content for the top-level `permutation_groups` file."""
    return f"""group: permutation_groups
task:
  - tc0_groups
  - nc1_groups
aggregate_metric_list:
{generate_aggregate_metric_list()}
metadata:
  version: 1.0
  description: Permutation composition benchmark for evaluating state-tracking capabilities
"""


def write(filename: str, content: str) -> None:
    """Write `content` to `filename` inside the benchmark directory."""
    (HERE / filename).write_text(content, encoding="utf-8")
    print(f"Generated {filename}")


def main() -> None:
    """Generate all task files."""
    tc0_groups: list[str] = []
    nc1_groups: list[str] = []

    for group_id, (group_name, group_desc, complexity) in GROUP_INFO.items():
        write(
            f"{group_id}_composition.yaml",
            generate_task_yaml(group_id, group_name, group_desc, complexity),
        )
        if complexity == "tc0":
            tc0_groups.append(group_id)
        else:
            nc1_groups.append(group_id)

    write("tc0_groups.yaml", generate_group_yaml("tc0", sorted(tc0_groups)))
    write("nc1_groups.yaml", generate_group_yaml("nc1", sorted(nc1_groups)))
    write("permutation_groups.yaml", generate_benchmark_yaml())

    print(f"\nTotal tasks generated: {len(GROUP_INFO)}")
    print(f"TC0 tasks: {len(tc0_groups)}")
    print(f"NC1 tasks: {len(nc1_groups)}")


if __name__ == "__main__":
    main()
