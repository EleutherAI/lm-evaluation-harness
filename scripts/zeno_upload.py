import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
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
    project = client.create_project(
        name=args.project_name,
        view="text-classification",
        metrics=[ZenoMetric(name="accuracy", type="mean", columns=["correct"])],
    )

    # Get all model subfolders from the parent data folder.
    models = [
        os.path.basename(os.path.normpath(f))
        for f in os.scandir(Path(args.data_path))
        if f.is_dir()
    ]

    assert len(models) == 0, "No model directories found in the data_path."

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
                "name": file.name.replace(model_args, "")
                .replace(".jsonl", "")
                .replace("_", ""),
                "file": file,
            }
            for file in files
        ]

        df = pd.DataFrame()
        system_df = pd.DataFrame()

        # Accumulate data for all tasks
        for index, task in enumerate(tasks):
            data = []
            with open(task["file"], "r") as file:
                for line in file:
                    data.append(json.loads(line))
            questions_answers = list(
                map(
                    lambda x: {
                        "question": x["arguments"][0][0],
                        "answers": list(map(lambda y: y[1], x["arguments"])),
                    },
                    data,
                )
            )
            if model_index == 0:  # Only need to assemble data for the first model
                df = (
                    generate_dataset(data, questions_answers, task["name"])
                    if index == 0
                    else pd.concat(
                        [df, generate_dataset(data, questions_answers, task["name"])]
                    )
                )
            system_df = (
                generate_system_df(data, questions_answers, task["name"])
                if index == 0
                else pd.concat(
                    [
                        system_df,
                        generate_system_df(data, questions_answers, task["name"]),
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
    questions_answers,
    task_name: str,
):
    """Generate a Zeno dataset from evaluation data.

    Args:
        data: The data to generate a dataset for.
        questions_answers: The questions and answers for all instances.
        task_name (str): The name of the task for which to format the data.

    Returns:
        pd.Dataframe: A dataframe that is ready to be uploaded to Zeno.
    """
    labels = list(map(lambda x: int(x["target"]), data))
    label_values = [
        questions_answers[index]["answers"][label] for index, label in enumerate(labels)
    ]
    df = pd.DataFrame(
        {
            "id": list(
                map(
                    lambda x: f"{task_name}_{str(x['doc_id'])}",
                    data,
                )
            ),
            "data": list(
                map(
                    lambda x: x["question"] + "\n\n" + "\n".join(x["answers"]),
                    questions_answers,
                )
            ),
            "task": task_name,
            "labels": label_values,
        }
    )
    return df


def generate_system_df(
    data,
    questions_answers,
    task_name: str,
):
    """Generate a dataframe for a specific system to be uploaded to Zeno.

    Args:
        data: The data to generate a dataframe from.
        questions_answers: The questions and answers for all instances.
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
    answers = []
    correct_list = []
    for element_index, element in enumerate(data):
        resps = list(map(lambda x: x[0], element["filtered_resps"]))
        if "acc_norm" in element:
            norm_logits = resps / np.array(
                [float(len(i)) for i in questions_answers[element_index]["answers"]]
            )
            answer = questions_answers[element_index]["answers"][np.argmax(norm_logits)]
            correct = (
                questions_answers[element_index]["answers"][element["target"]] == answer
            )
            answer = (
                answer
                + "\n\n"
                + "Raw Pred.: "
                + ", ".join(map(lambda y: str(round(y, 2)), resps))
                + "\n\n"
                + "Norm Pred.: "
                + ", ".join(map(lambda y: str(round(y, 2)), norm_logits))
            )
        else:
            answer = questions_answers[element_index]["answers"][np.argmax(resps)]
            correct = (
                questions_answers[element_index]["answers"][element["target"]] == answer
            )
            answer = (
                answer
                + "\n\n"
                + "Pred.: "
                + ", ".join(map(lambda y: str(round(y, 2)), resps))
            )
        answers.append(answer)
        correct_list.append(correct)
    system_df = pd.DataFrame({"id": ids, "output": answers, "correct": correct_list})
    return system_df


if __name__ == "__main__":
    main()
