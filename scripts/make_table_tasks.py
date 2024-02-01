"""
Usage:
   Writes csv and Markdown table to csv_file, md_file (below).
"""
import logging
import os
from pathlib import Path
from typing import List, Union

import datasets
import pandas as pd

from lm_eval import tasks
from lm_eval.utils import load_yaml_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
datasets.disable_caching()
task_manager = tasks.TaskManager


def load_changed_files(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        content = f.read()
        words_list = [x for x in content.split()]
    return words_list


def parser(full_path: List[str]) -> List[str]:
    _output = set()
    for x in full_path:
        if x.endswith(".yaml"):
            _output.add(load_yaml_config(x)["task"])
        elif x.endswith(".py"):
            path = [str(x) for x in (list(Path(x).parent.glob("*.yaml")))]
            _output |= {load_yaml_config(x)["task"] for x in path}
    return list(_output)


def new_tasks() -> Union[List[str], None]:
    FILENAME = ".github/outputs/tasks_all_changed_and_modified_files.txt"
    if os.path.exists(FILENAME):
        # If tasks folder has changed then we get the list of files from FILENAME
        # and parse the yaml files to get the task names.
        return parser(load_changed_files(FILENAME))
    elif os.getenv("API") is not None:
        # Or if API has changed then we set the ENV variable API to True
        # and run  given tasks.
        return ["arc_easy", "hellaswag", "piqa", "wikitext"]
    # if both not true just do arc_easy
    else:
        return None


def check(tf):
    if tf:
        return "✓"
    else:
        return " "


def maketable(df):
    headers = [
        "Task Name",
        "Group",
        "Train",
        "Val",
        "Test",
        "Val/Test Docs",
        "Request Type,",
        "Metrics",
    ]
    values = []
    if not df:
        _tasks = task_manager.TASK_REGISTRY.items()
        _tasks = sorted(_tasks, key=lambda x: x[0])
    else:
        task_classes = new_tasks()
        _tasks = [(x, task_manager.TASK_REGISTRY.get(x)) for x in task_classes]
    count = 0
    for tname, Task in _tasks:
        task = Task()
        v = [
            tname,
            task.config.group,
            check(task.has_training_docs()),
            check(task.has_validation_docs()),
            check(task.has_test_docs()),
            len(
                list(
                    task.test_docs()
                    if task.has_test_docs()
                    else task.validation_docs()
                    if task.has_validation_docs()
                    else task.training_docs()
                )
            ),
            task.config.output_type,
            ", ".join(task.aggregation().keys()),
        ]
        logger.info(v)
        values.append(v)
        count += 1
        if count == 10:
            break
    if not df:
        df = pd.DataFrame(values, columns=headers)
        table = df.to_markdown(index=False)
    else:
        for new_row in values:
            tname = new_row[0]
            if tname in df["Task Name"].values:
                # If task name exists, update the row
                df.loc[df["Task Name"] == tname] = new_row
            else:
                # If task name doesn't exist, append a new row
                series = pd.Series(new_row, index=df.columns)
                df = pd.concat([df, series.to_frame().T], ignore_index=True)
        df = df.sort_values(by=["Task Name"])
        table = df.to_markdown(index=False)
    return df, table


if __name__ == "__main__":
    csv_file = Path(f"{Path(__file__).parent.parent.resolve()}/docs/task_table.csv")
    md_file = Path(f"{Path(__file__).parent.parent.resolve()}/docs/task_table.md")

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = None

    df, table = maketable(df=df)

    with open(md_file, "w") as f:
        f.write(table)
    with open(csv_file, "w") as f:
        df.to_csv(f, index=False)
