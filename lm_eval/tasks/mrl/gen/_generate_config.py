from pathlib import Path

import datasets


if __name__ == "__main__":
    subsets = [
        x
        for x in datasets.get_dataset_config_names(
            "mrlbenchmarks/global-piqa-nonparallel"
        )
        if not x.startswith("dev")
    ]
    PARENT = Path(__file__).parent
    for s in subsets:
        with open(PARENT / f"{s}.yaml", "w") as f:
            f.write("include: '_template'\n")
            f.write(f"task: {s}\n")
            f.write(f"dataset_name: {s}\n")

    with open(PARENT / "_global_piqa_gen.yaml", "w") as f:
        f.write("group: global_piqa_gen\n")
        f.write("task:\n")
        for s in subsets:
            f.write(f"  - task: {s}\n")
        f.write("aggregate_metric_list:\n")
        f.write("  - metric: exact_match\n")
        f.write("    aggregation: mean\n")
        f.write("    weight_by_size: true\n")
        f.write("metadata:\n")
        f.write("  version: 1.0\n")
