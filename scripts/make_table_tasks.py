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
from tqdm import tqdm

from lm_eval import tasks
from lm_eval.utils import load_yaml_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
datasets.disable_caching()
task_manager = tasks.TaskManager()


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


def new_tasks(df=None) -> Union[List[str], None]:
    new_tasks = []
    FILENAME = ".github/outputs/tasks_all_changed_and_modified_files.txt"
    if os.path.exists(FILENAME):
        # If tasks folder has changed then we get the list of files from FILENAME
        # and parse the yaml files to get the task names.
        # (for when run in CI)
        new_tasks.extend(parser(load_changed_files(FILENAME)))
    # if we already have a (partial) task table created, only add tasks
    # which aren't already in task table
    if df is not None:
        _tasks = task_manager.all_tasks
        _tasks = [k for k in _tasks if k not in df["Task Name"].values]

        new_tasks.extend(_tasks)
    # if both not true just do arc_easy
    return new_tasks


def check(tf):
    if tf:
        return "✓"
    else:
        return " "


def maketable(df):
    headers = [
        # For now, we restrict to presenting data
        # That can be collected statically.
        "Task Name",
        "Group",
        # "Train",
        # "Val",
        # "Test",
        # "Val/Test Docs",
        "Request Type",
        "Filters",
        "Metrics",
    ]
    values = []
    if df is None:
        _tasks = task_manager.all_tasks
    else:
        _tasks = new_tasks(df=df)

    for tname in tqdm(_tasks):
        task_config = task_manager._get_config(tname)
        if not task_config:
            continue
        # TODO: also catch benchmark configs like flan
        if not isinstance(task_config["task"], str):
            continue
        if task_config.get("class", None):
            continue
        v = [
            tname,
            task_config.get("group", None),
            task_config.get("output_type", "greedy_until"),
            ", ".join(
                str(f["name"])
                for f in task_config.get("filter_list", [{"name": "none"}])
            ),
            ", ".join(str(metric["metric"]) for metric in task_config["metric_list"]),
        ]

        logger.info(v)
        values.append(v)

    if df is None:
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
    df = None
    df, table = maketable(df=df)

    with open(md_file, "w") as f:
        f.write(table)
    with open(csv_file, "w") as f:
        df.to_csv(f, index=False)
