from pathlib import Path

import datasets
import yaml


class IndentedDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentedDumper, self).increase_indent(flow, False)


PREFACE = "global_piqa_completions"


def format_subset(subset: str, preface: str = PREFACE) -> str:
    return f"{preface}_{subset}"


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
            yaml.dump(
                {
                    "include": "_template",
                    "task": format_subset(s),
                    "dataset_name": s,
                },
                f,
            )

    with open(PARENT / "_global_piqa.yaml", "w") as f:
        yaml.dump(
            {
                "group": f"{PREFACE}",
                "task": [{"task": format_subset(s), "task_alias": s} for s in subsets],
                "aggregate_metric_list": [
                    {"metric": m, "aggregation": "mean", "weight_by_size": True}
                    for m in ["acc", "acc_norm", "acc_bytes"]
                ],
            },
            f,
            Dumper=IndentedDumper,
            default_flow_style=False,
            sort_keys=False,
        )
        f.write("metadata:\n")
        f.write("  version: 1.0\n")
