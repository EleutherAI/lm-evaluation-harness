"""Generate per-subtask YAML configs for BBEH (BIG-Bench Extra Hard).

All 23 subtasks live in a single flat split of the official ``BBEH/bbeh``
dataset, distinguished by the ``task`` column. Each generated config selects its
subtask with ``process_docs: !function utils.process_<subtask>`` (the matching
``partial`` filters live in ``utils.py``). Run from this directory:

    python _generate_configs.py
"""

import os


# The 23 BBEH subtasks. The official `task` column uses spaces, so the matching
# filter for "boardgame_qa" is utils.process_boardgame_qa (task_name="boardgame qa").
SUBTASKS = [
    "boardgame_qa",
    "boolean_expressions",
    "buggy_tables",
    "causal_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "geometric_shapes",
    "hyperbaton",
    "linguini",
    "movie_recommendation",
    "multistep_arithmetic",
    "nycc",
    "object_counting",
    "object_properties",
    "sarc_triples",
    "shuffled_objects",
    "spatial_reasoning",
    "sportqa",
    "temporal_sequence",
    "time_arithmetic",
    "web_of_lies",
    "word_sorting",
    "zebra_puzzles",
]

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    # Per-subtask task configs.
    for subtask in SUBTASKS:
        path = os.path.join(HERE, f"bbeh_{subtask}.yaml")
        with open(path, "w") as f:
            f.write('"include": "_bbeh_template_yaml"\n')
            f.write(f'"task": "bbeh_{subtask}"\n')
            f.write(f'"process_docs": !function utils.process_{subtask}\n')

    # Group config aggregating all subtasks (size-weighted mean = micro average).
    group_path = os.path.join(HERE, "_bbeh.yaml")
    with open(group_path, "w") as f:
        f.write("group: bbeh\n")
        f.write("task:\n")
        for subtask in SUBTASKS:
            f.write(f"  - bbeh_{subtask}\n")
        f.write("aggregate_metric_list:\n")
        f.write("  - metric: exact_match\n")
        f.write("    aggregation: mean\n")
        f.write("    weight_by_size: true\n")
        f.write("metadata:\n")
        f.write("  version: 1.0\n")

    print(f"wrote {len(SUBTASKS)} subtask configs + _bbeh.yaml")


if __name__ == "__main__":
    main()
