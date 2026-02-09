#!/usr/bin/env python3
"""Generate all task YAML files for the permutation benchmark."""

from typing import List


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
    "z2_1": ("Z2¹", "elementary abelian group Z2", "tc0"),
    "z2_2": ("Z2²", "elementary abelian group Z2²", "tc0"),
    "z2_3": ("Z2³", "elementary abelian group Z2³", "tc0"),
    "z2_4": ("Z2⁴", "elementary abelian group Z2⁴", "tc0"),
    "z2_5": ("Z2⁵", "elementary abelian group Z2⁵", "tc0"),
    "z3_1": ("Z3¹", "elementary abelian group Z3", "tc0"),
    "z3_2": ("Z3²", "elementary abelian group Z3²", "tc0"),
    "z3_3": ("Z3³", "elementary abelian group Z3³", "tc0"),
    "z3_4": ("Z3⁴", "elementary abelian group Z3⁴", "tc0"),
    "z5_1": ("Z5¹", "elementary abelian group Z5", "tc0"),
    "z5_2": ("Z5²", "elementary abelian group Z5²", "tc0"),
    "z5_3": ("Z5³", "elementary abelian group Z5³", "tc0"),
    "z5_4": ("Z5⁴", "elementary abelian group Z5⁴", "tc0"),
    "psl2_2": ("PSL(2,2)", "projective special linear group PSL(2,2)", "tc0"),
    "psl2_3": ("PSL(2,3)", "projective special linear group PSL(2,3)", "tc0"),
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
    "psl2_4": ("PSL(2,4)", "projective special linear group PSL(2,4)", "nc1"),
    "psl2_5": ("PSL(2,5)", "projective special linear group PSL(2,5)", "nc1"),
    "psl2_7": ("PSL(2,7)", "projective special linear group PSL(2,7)", "nc1"),
    "psl2_8": ("PSL(2,8)", "projective special linear group PSL(2,8)", "nc1"),
    "psl2_9": ("PSL(2,9)", "projective special linear group PSL(2,9)", "nc1"),
    "psl2_11": ("PSL(2,11)", "projective special linear group PSL(2,11)", "nc1"),
    "psl3_2": ("PSL(3,2)", "projective special linear group PSL(3,2)", "nc1"),
    "psl3_3": ("PSL(3,3)", "projective special linear group PSL(3,3)", "nc1"),
    "psl3_4": ("PSL(3,4)", "projective special linear group PSL(3,4)", "nc1"),
    "psl3_5": ("PSL(3,5)", "projective special linear group PSL(3,5)", "nc1"),
    "m11": ("M11", "Mathieu group M11", "nc1"),
    "m12": ("M12", "Mathieu group M12", "nc1"),
}


def generate_metric_list() -> str:
    """Generate the metric list for all sequence lengths."""
    metrics = []
    # Generate all 100 metrics from 5 to 500 in increments of 5
    for length in range(5, 505, 5):
        metrics.append(f"""  - metric: "{length}"
    aggregation: !function group_composition_utils.aggregate_metrics
    higher_is_better: true""")
    return "\n".join(metrics)


def generate_task_yaml(
    group_id: str, group_name: str, group_desc: str, complexity: str
) -> str:
    """Generate YAML content for a single task."""
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
doc_to_text: |
  You are given a sequence of permutations from the group {group_name} ({group_desc}), identified by their integer IDs. Your task is to compute their composed product.

  The composition must be performed sequentially from right to left, following the standard mathematical convention (p_n ∘ ... ∘ p_2 ∘ p_1).

  Sequence: {{{{input_sequence}}}}

  Question: What is the integer ID of the final composed permutation?

  Answer:
doc_to_target: "{{{{target}}}}"
process_results: !function group_composition_utils.process_results
metric_list:
{generate_metric_list()}
metadata:
  version: 1.0"""


def generate_group_yaml(name: str, groups: List[str]) -> str:
    """Generate YAML content for a group file (TC0 or NC1)."""
    tasks = [f"  - {group}_composition" for group in groups]

    # Generate all 100 aggregate metrics
    aggregate_metrics = []
    for length in range(5, 505, 5):
        aggregate_metrics.append(f"""  - metric: "{length}"
    weight_by_size: false""")

    return f"""group: {name}_groups
task:
{chr(10).join(tasks)}
aggregate_metric_list:
{chr(10).join(aggregate_metrics)}
metadata:
  version: 1.0
  description: Permutation composition tasks for {name.upper()} complexity class"""


def main():
    """Generate all task files."""
    # Create lists for TC0 and NC1 groups
    tc0_groups = []
    nc1_groups = []

    # Generate individual task files
    for group_id, (group_name, group_desc, complexity) in GROUP_INFO.items():
        filename = f"{group_id}_composition.yaml"
        content = generate_task_yaml(group_id, group_name, group_desc, complexity)

        print(f"Generated {filename}")

        # Write the file
        with open(filename, "w") as f:
            f.write(content)

        # Add to appropriate list
        if complexity == "tc0":
            tc0_groups.append(group_id)
        else:
            nc1_groups.append(group_id)

    # Generate group files
    tc0_content = generate_group_yaml("tc0", sorted(tc0_groups))
    nc1_content = generate_group_yaml("nc1", sorted(nc1_groups))

    print("\nGenerated tc0_groups.yaml")
    print("Generated nc1_groups.yaml")

    # Generate main group file with all 100 aggregate metrics
    aggregate_metrics = []
    for length in range(5, 505, 5):
        aggregate_metrics.append(f"""  - metric: "{length}"
    weight_by_size: false""")

    main_content = f"""group: permutation_groups
task:
  - tc0_groups
  - nc1_groups
aggregate_metric_list:
{chr(10).join(aggregate_metrics)}
metadata:
  version: 1.0
  description: Permutation composition benchmark for evaluating state-tracking capabilities"""

    print("Generated permutation_groups.yaml")

    # Write files
    with open("tc0_groups.yaml", "w") as f:
        f.write(tc0_content)
    with open("nc1_groups.yaml", "w") as f:
        f.write(nc1_content)
    with open("permutation_groups.yaml", "w") as f:
        f.write(main_content)

    print(f"\nTotal tasks generated: {len(GROUP_INFO)}")
    print(f"TC0 tasks: {len(tc0_groups)}")
    print(f"NC1 tasks: {len(nc1_groups)}")


if __name__ == "__main__":
    main()
