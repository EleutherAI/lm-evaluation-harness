import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from zeno_client import ZenoClient


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

    client = ZenoClient(os.environ["ZENO_API_KEY"], endpoint="http://localhost:8000")
    project = client.create_project(
        name=args.project_name,
        view="text-classification",
        metrics=[],  # TODO: Check metrics
    )

    # Get all model subfolders from the parent data folder.
    models = [
        os.path.basename(os.path.normpath(f))
        for f in os.scandir(Path(args.data_path))
        if f.is_dir()
    ]

    assert len(models) > 0, "No model directories found in the data_path."

    # Upload data for all models
    for model_index, model in enumerate(models):
        # Get all output files for the tasks
        dir_path = Path(args.data_path, model)

        # Task files are saved with model args as a prefix. We don't want this for the Zeno task names.
        model_args = re.sub(
            "/|=",
            "__",
            json.load(open(Path(dir_path, "results.json")))["config"]["model_args"],
        )
        files = list(dir_path.glob("*.jsonl"))
        tasks = [
            {
                "name": file.name.replace(model_args, "").replace(".jsonl", "")[1:],
                "file": file,
            }
            for file in files
        ]

        df = pd.DataFrame()
        system_df = pd.DataFrame()

        configs = json.load(open(Path(dir_path, "results.json")))["configs"]

        # Accumulate data for all tasks
        for index, task in enumerate(tasks):
            config = configs[task["name"]]
            data = []
            with open(task["file"], "r") as file:
                data = json.loads(file.read())

            if model_index == 0:  # Only need to assemble data for the first model
                df = (
                    generate_dataset(data, config, task["name"])
                    if index == 0
                    else pd.concat([df, generate_dataset(data, config, task["name"])])
                )

            system_df = (
                generate_system_df(data, config, task["name"])
                if index == 0
                else pd.concat(
                    [
                        system_df,
                        generate_system_df(data, config, task["name"]),
                    ]
                )
            )

        if model_index == 0:  # Only need to upload data for the first model
            project.upload_dataset(
                df, id_column="id", data_column="data", label_column="labels"
            )

        project.upload_system(
            system_df,
            name=model,
            id_column="id",
            output_column="output",
        )


def generate_dataset(
    data,
    config,
    task_name: str,
):
    """Generate a Zeno dataset from evaluation data.

    Args:
        data: The data to generate a dataset for.
        config: The configuration of the task.
        task_name (str): The name of the task for which to format the data.

    Returns:
        pd.Dataframe: A dataframe that is ready to be uploaded to Zeno.
    """
    ids = list(
        map(
            lambda x: f"{task_name}_{str(x['doc_id'])}",
            data,
        )
    )
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
            "task": task_name,
            "labels": labels,
            "output_type": config["output_type"],
        }
    )


def generate_system_df(
    data,
    config,
    task_name: str,
):
    """Generate a dataframe for a specific system to be uploaded to Zeno.

    Args:
        data: The data to generate a dataframe from.
        config: The configuration of the task.
        task_name (str): The name of the task for which to format the data.

    Returns:
        pd.Dataframe: A dataframe that is ready to be uploaded to Zeno as a system.
    """
    ids = list(
        map(
            lambda x: f"{task_name}_{str(x['doc_id'])}",
            data,
        )
    )
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
            metrics[metric["metric"]] = list(
                map(lambda x: str(x[metric["metric"]]), data)
            )

    system_dict = {"id": ids, "output": answers}
    system_dict.update(metrics)
    system_df = pd.DataFrame(system_dict)
    return system_df


if __name__ == "__main__":
    main()
