import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from zeno_client import ZenoClient, ZenoMetric


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

    tasks = tasks_for_model(models[0], args.data_path)

    for model in models:  # Make sure that all models have the same tasks.
        assert (
            tasks_for_model(model, args.data_path) == tasks
        ), "All models must have the same tasks."

    for task in tasks:
        # Upload data for all models
        for model_index, model in enumerate(models):
            model_args = re.sub(
                "/|=",
                "__",
                json.load(open(Path(args.data_path, model, "results.json")))["config"][
                    "model_args"
                ],
            )
            with open(
                Path(args.data_path, model, f"{model_args}_{task}.jsonl"), "r"
            ) as file:
                data = json.loads(file.read())

            configs = json.load(open(Path(args.data_path, model, "results.json")))[
                "configs"
            ]
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
    config = (json.load(open(Path(dir_path, "results.json")))["configs"],)
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
    ids = list(map(lambda x: str(x["doc_id"]), data))
    labels = list(map(lambda x: str(x["target"]), data))
    instance = [""] * len(ids)

    if config["output_type"] == "loglikelihood":
        instance = list(map(lambda x: str(x["arguments"][0][0]), data))
        labels = list(map(lambda x: str(x["arguments"][0][1]), data))
    elif config["output_type"] == "multiple_choice":
        instance = list(
            map(
                lambda x: str(
                    x["arguments"][0][0]
                    + "\n\n"
                    + "\n".join(list(map(lambda y: f"- {y[1]}", x["arguments"])))
                ),
                data,
            )
        )
    elif config["output_type"] == "loglikelihood_rolling":
        instance = list(map(lambda x: str(x["arguments"][0][0]), data))
    elif config["output_type"] == "generate_until":
        instance = list(map(lambda x: str(x["arguments"][0][0]), data))

    return pd.DataFrame(
        {
            "id": ids,
            "data": instance,
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
    ids = list(map(lambda x: str(x["doc_id"]), data))
    answers = [""] * len(ids)

    if config["output_type"] == "loglikelihood":
        answers = list(
            map(
                lambda x: "correct"
                if x["filtered_resps"][0][1] == True
                else "incorrect",
                data,
            )
        )
    elif config["output_type"] == "multiple_choice":
        answers = list(
            map(
                lambda x: ", ".join(
                    list(map(lambda y: str(y[0]), x["filtered_resps"]))
                ),
                data,
            )
        )
    elif config["output_type"] == "loglikelihood_rolling":
        answers = list(map(lambda x: str(x["filtered_resps"][0]), data))
    elif config["output_type"] == "generate_until":
        answers = list(map(lambda x: str(x["filtered_resps"][0]), data))

    metrics = {}
    for metric in config["metric_list"]:
        if "aggregation" in metric and metric["aggregation"] == "mean":
            metrics[metric["metric"]] = list(map(lambda x: x[metric["metric"]], data))

    system_dict = {"id": ids, "output": answers}
    system_dict.update(metrics)
    system_df = pd.DataFrame(system_dict)
    return system_df


if __name__ == "__main__":
    main()
