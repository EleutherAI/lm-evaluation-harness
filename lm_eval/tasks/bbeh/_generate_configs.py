"""Generate the pinned BBEH task and group configurations."""

from pathlib import Path

import yaml


SOURCE_REVISION = "80d12ca916b7158f22293fcf3144f4d3d854d4be"
RAW_ROOT = f"https://raw.githubusercontent.com/google-deepmind/bbeh/{SOURCE_REVISION}"
TASKS = (
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
)


def task_config(task: str) -> dict:
    return {
        "include": "_default_template_yaml",
        "task": f"bbeh_{task}",
        "task_alias": task,
        "dataset_kwargs": {
            "data_files": {
                "test": (f"{RAW_ROOT}/bbeh/benchmark_tasks/bbeh_{task}/task.json")
            },
            "field": "examples",
        },
    }


def group_config() -> dict:
    return {
        "group": "bbeh",
        "group_alias": "BBEH",
        "task": [f"bbeh_{task}" for task in TASKS],
        "aggregate_metric_list": [
            {
                "metric": "bbeh_acc",
                "aggregation": "mean",
                "weight_by_size": True,
            }
        ],
        "metadata": {
            "version": 1.0,
            "source_revision": SOURCE_REVISION,
            "headline_metric_note": (
                "Compute the official harmonic mean from the 23 task rows; "
                "the group value is micro accuracy."
            ),
        },
    }


def main() -> None:
    root = Path(__file__).parent
    for task in TASKS:
        path = root / f"bbeh_{task}.yaml"
        path.write_text(
            yaml.safe_dump(task_config(task), sort_keys=False), encoding="utf-8"
        )
    (root / "_bbeh.yaml").write_text(
        yaml.safe_dump(group_config(), sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
