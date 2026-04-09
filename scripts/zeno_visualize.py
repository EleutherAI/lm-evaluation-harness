import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Union

import pandas as pd
from zeno_client import ZenoClient, ZenoMetric

from lm_eval.utils import (
    get_file_task_name,
    get_latest_filename,
    get_results_filenames,
    get_sample_results_filenames,
    sanitize_model_name,
)


eval_logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload your data to the Zeno AI evaluation platform to visualize results. This requires a ZENO_API_KEY in your environment variables. The eleuther harness must be run with log_samples=True and an output_path set for data to be written to disk."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Where to find the results of the benchmarks that have been run. Uses the name of each subfolder as the model name.",
    )
    parser.add_argument(
        "--project_name",
        required=True,
        help="The name of the generated Zeno project.",
    )
    return parser.parse_args()


def sanitize_string(model_args_raw: Union[str, dict]) -> str:
    """Sanitize the model_args string or dict"""
    # Convert to string if it's a dictionary
    model_args_str = (
        json.dumps(model_args_raw)
        if isinstance(model_args_raw, dict)
        else model_args_raw
    )
    # Apply the sanitization
    return re.sub(
        r"[\"<>:/|\\?*\[\]]+",
        "__",
        model_args_str,
    )


def get_model_files(model_dir: Path) -> list[str]:
    """Return the files that live directly inside a candidate model directory."""
    return [path.as_posix() for path in model_dir.iterdir() if path.is_file()]


def discover_model_dirs(data_path: Union[str, Path]) -> list[Path]:
    """Find directories under data_path that contain aggregated result files."""
    data_root = Path(data_path)
    candidate_dirs = [data_root] + sorted(
        (path for path in data_root.rglob("*") if path.is_dir()),
        key=lambda path: path.as_posix(),
    )
    model_dirs = [
        path for path in candidate_dirs if get_results_filenames(get_model_files(path))
    ]

    if not model_dirs:
        raise ValueError(
            f"No model result directories found under {data_root}. "
            "Expected folders containing results_*.json files."
        )

    return model_dirs


def get_model_name(model_dir: Path, data_root: Path) -> str:
    """Build a Zeno-safe display name for a model directory under data_root."""
    relative_model_dir = model_dir.relative_to(data_root)
    model_name = (
        relative_model_dir.as_posix()
        if relative_model_dir != Path(".")
        else model_dir.name
    )
    return sanitize_model_name(model_name)


def get_latest_results_file(model_dir: Path) -> str:
    """Return the latest aggregated results file for a model directory."""
    model_results_filenames = get_results_filenames(get_model_files(model_dir))
    return get_latest_filename(model_results_filenames)


def get_latest_sample_results_file(model_dir: Path, task: str) -> str:
    """Return the latest sample results file for a task in a model directory."""
    model_sample_filenames = [
        path
        for path in get_sample_results_filenames(get_model_files(model_dir))
        if get_file_task_name(Path(path).name) == task
    ]
    return get_latest_filename(model_sample_filenames)


def main():
    """Upload the results of your benchmark tasks to the Zeno AI evaluation platform.

    This scripts expects your results to live in a data folder where subfolders contain results of individual models.
    """
    args = parse_args()

    client = ZenoClient(os.environ["ZENO_API_KEY"])

    data_root = Path(args.data_path)
    model_dirs = discover_model_dirs(data_root)
    model_names = {
        model_dir: get_model_name(model_dir, data_root) for model_dir in model_dirs
    }

    # Get the tasks from the latest results file of the first model.
    tasks = set(tasks_for_model(model_dirs[0]))

    # Get tasks names from the latest results file for each model
    # Get intersection of tasks for all models
    for model_dir in model_dirs:
        old_tasks = tasks.copy()
        task_count = len(tasks)
        model_tasks = set(tasks_for_model(model_dir))
        tasks.intersection(set(model_tasks))

        if task_count != len(tasks):
            eval_logger.warning(
                f"All models must have the same tasks. {model_names[model_dir]} has tasks: {model_tasks} but have already recorded tasks: {old_tasks}. Taking intersection {tasks}"
            )

    assert len(tasks) > 0, (
        "Must provide at least one task in common amongst models to compare."
    )

    for task in tasks:
        # Upload data for all models
        for model_index, model_dir in enumerate(model_dirs):
            model = model_names[model_dir]
            # Get latest results and sample results for a model
            latest_results = get_latest_results_file(model_dir)
            latest_sample_results = get_latest_sample_results_file(model_dir, task)
            # Load the model_args, which can be either a string or a dictionary
            model_args = sanitize_string(
                json.load(
                    open(
                        latest_results,
                        encoding="utf-8",
                    )
                )["config"]["model_args"]
            )

            print(model_args)
            data = []
            with open(
                latest_sample_results,
                "r",
                encoding="utf-8",
            ) as file:
                for line in file:
                    data.append(json.loads(line.strip()))

            configs = json.load(
                open(latest_results, encoding="utf-8")
            )["configs"]
            config = configs[task]

            if model_index == 0:  # Only need to assemble data for the first model
                metrics = []
                for metric in config["metric_list"]:
                    if metric.get("aggregation") == "mean":
                        metrics.append(
                            ZenoMetric(
                                name=metric["metric"],
                                type="mean",
                                columns=[metric["metric"]],
                            )
                        )
                project = client.create_project(
                    name=args.project_name + (f"_{task}" if len(tasks) > 1 else ""),
                    view="text-classification",
                    metrics=metrics,
                )
                project.upload_dataset(
                    generate_dataset(data, config),
                    id_column="id",
                    data_column="data",
                    label_column="labels",
                )

            project.upload_system(
                generate_system_df(data, config),
                name=model,
                id_column="id",
                output_column="output",
            )


def tasks_for_model(model_dir: Path):
    """Get the tasks for a specific model.

    Args:
        model_dir (Path): The path to the model results directory.

    Returns:
        list: A list of tasks for the model.
    """
    # get latest model results for a given name
    latest_results = get_latest_results_file(model_dir)
    config = (json.load(open(latest_results, encoding="utf-8"))["configs"],)
    return list(config[0].keys())


def generate_dataset(
    data,
    config,
):
    """Generate a Zeno dataset from evaluation data.

    Args:
        data: The data to generate a dataset for.
        config: The configuration of the task.

    Returns:
        pd.Dataframe: A dataframe that is ready to be uploaded to Zeno.
    """
    ids = (
        [x["doc_id"] for x in data]
        if not config.get("filter_list")
        else [f"{x['doc_id']}.{x['filter']}" for x in data]
    )
    labels = [x["target"] for x in data]
    instance = [""] * len(ids)

    if config["output_type"] == "loglikelihood":
        instance = [x["arguments"]["gen_args_0"]["arg_0"] for x in data]
        labels = [x["arguments"]["gen_args_0"]["arg_1"] for x in data]
    elif config["output_type"] == "multiple_choice":
        instance = [
            x["arguments"]["gen_args_0"]["arg_0"]
            + "\n\n"
            + "\n".join([f"- {y[1]}" for y in x["arguments"]])
            for x in data
        ]
    elif config["output_type"] == "loglikelihood_rolling":
        instance = [x["arguments"]["gen_args_0"]["arg_0"] for x in data]
    elif config["output_type"] == "generate_until":
        instance = [x["arguments"]["gen_args_0"]["arg_0"] for x in data]

    return pd.DataFrame(
        {
            "id": ids,
            "doc_id": [x["doc_id"] for x in data],
            "data": instance,
            "input_len": [len(x) for x in instance],
            "labels": labels,
            "output_type": config["output_type"],
        }
    )


def generate_system_df(data, config):
    """Generate a dataframe for a specific system to be uploaded to Zeno.

    Args:
        data: The data to generate a dataframe from.
        config: The configuration of the task.

    Returns:
        pd.Dataframe: A dataframe that is ready to be uploaded to Zeno as a system.
    """
    ids = (
        [x["doc_id"] for x in data]
        if not config.get("filter_list")
        else [f"{x['doc_id']}.{x['filter']}" for x in data]
    )
    system_dict = {"id": ids}
    system_dict["doc_id"] = [x["doc_id"] for x in data]
    if config.get("filter_list"):
        system_dict["filter"] = [x["filter"] for x in data]
    system_dict["output"] = [""] * len(ids)

    if config["output_type"] == "loglikelihood":
        system_dict["output"] = [
            "correct" if x["filtered_resps"][0][1] is True else "incorrect"
            for x in data
        ]
    elif config["output_type"] == "multiple_choice":
        system_dict["output"] = [
            ", ".join([str(y[0]) for y in x["filtered_resps"]]) for x in data
        ]
        system_dict["num_answers"] = [len(x["filtered_resps"]) for x in data]
    elif config["output_type"] == "loglikelihood_rolling":
        system_dict["output"] = [str(x["filtered_resps"][0]) for x in data]
    elif config["output_type"] == "generate_until":
        system_dict["output"] = [str(x["filtered_resps"][0]) for x in data]
        system_dict["output_length"] = [len(str(x["filtered_resps"][0])) for x in data]

    metrics = {
        metric["metric"]: [x[metric["metric"]] for x in data]
        for metric in config["metric_list"]
    }
    system_dict.update(metrics)
    system_df = pd.DataFrame(system_dict)
    return system_df


if __name__ == "__main__":
    main()
