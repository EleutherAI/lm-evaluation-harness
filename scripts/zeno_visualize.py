import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from zeno_client import ZenoClient, ZenoMetric

from lm_eval.utils import eval_logger


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


def main():
    """Upload the results of your benchmark tasks to the Zeno AI evaluation platform.

    This scripts expects your results to live in a data folder where subfolders contain results of individual models.
    """
    args = parse_args()

    client = ZenoClient(os.environ["ZENO_API_KEY"])

    # Get all model subfolders from the parent data folder.
    models = [
        os.path.basename(os.path.normpath(f))
        for f in os.scandir(Path(args.data_path))
        if f.is_dir()
    ]

    assert len(models) > 0, "No model directories found in the data_path."

    tasks = set(tasks_for_model(models[0], args.data_path))

    for model in models:  # Make sure that all models have the same tasks.
        old_tasks = tasks.copy()
        task_count = len(tasks)

        model_tasks = tasks_for_model(model, args.data_path)
        tasks.intersection(set(model_tasks))

        if task_count != len(tasks):
            eval_logger.warning(
                f"All models must have the same tasks. {model} has tasks: {model_tasks} but have already recorded tasks: {old_tasks}. Taking intersection {tasks}"
            )

    assert (
        len(tasks) > 0
    ), "Must provide at least one task in common amongst models to compare."

    for task in tasks:
        # Upload data for all models
        for model_index, model in enumerate(models):
            model_args = re.sub(
                "/|=",
                "__",
                json.load(
                    open(Path(args.data_path, model, "results.json"), encoding="utf-8")
                )["config"]["model_args"],
            )
            with open(
                Path(args.data_path, model, f"{model_args}_{task}.jsonl"),
                "r",
                encoding="utf-8",
            ) as file:
                data = json.loads(file.read())

            configs = json.load(
                open(Path(args.data_path, model, "results.json"), encoding="utf-8")
            )["configs"]
            config = configs[task]

            if model_index == 0:  # Only need to assemble data for the first model
                metrics = []
                for metric in config["metric_list"]:
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


def tasks_for_model(model: str, data_path: str):
    """Get the tasks for a specific model.

    Args:
        model (str): The name of the model.
        data_path (str): The path to the data.

    Returns:
        list: A list of tasks for the model.
    """
    dir_path = Path(data_path, model)
    config = (
        json.load(open(Path(dir_path, "results.json"), encoding="utf-8"))["configs"],
    )
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
    ids = [x["doc_id"] for x in data]
    labels = [x["target"] for x in data]
    instance = [""] * len(ids)

    if config["output_type"] == "loglikelihood":
        instance = [x["arguments"][0][0] for x in data]
        labels = [x["arguments"][0][1] for x in data]
    elif config["output_type"] == "multiple_choice":
        instance = [
            x["arguments"][0][0]
            + "\n\n"
            + "\n".join([f"- {y[1]}" for y in x["arguments"]])
            for x in data
        ]
    elif config["output_type"] == "loglikelihood_rolling":
        instance = [x["arguments"][0][0] for x in data]
    elif config["output_type"] == "generate_until":
        instance = [x["arguments"][0][0] for x in data]

    return pd.DataFrame(
        {
            "id": ids,
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
    ids = [x["doc_id"] for x in data]
    system_dict = {"id": ids}
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

    metrics = {}
    for metric in config["metric_list"]:
        if "aggregation" in metric and metric["aggregation"] == "mean":
            metrics[metric["metric"]] = [x[metric["metric"]] for x in data]

    system_dict.update(metrics)
    system_df = pd.DataFrame(system_dict)
    return system_df


if __name__ == "__main__":
    main()
