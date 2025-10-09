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
            f.write("include: '_template_mc'\n")
            f.write(f"task: mrl_{s}\n")
            f.write(f"dataset_name: {s}\n")

with open(PARENT / "_global_piqa.yaml", "w") as f:
    f.write("group: global_piqa\n")
    f.write("task:\n")
    for s in subsets:
        f.write(f"  - task: mrl_{s}\n")
        f.write(f"    task_alias: {s}\n")
    f.write("aggregate_metric_list:\n")
    f.write("  - metric: acc\n")
    f.write("    aggregation: mean\n")
    f.write("    weight_by_size: true\n")
    f.write("  - metric: acc_norm\n")
    f.write("    aggregation: mean\n")
    f.write("    weight_by_size: true\n")
    f.write("  - metric: acc_bytes\n")
    f.write("    aggregation: mean\n")
    f.write("    weight_by_size: true\n")
    f.write("metadata:\n")
    f.write("  version: 1.0\n")
