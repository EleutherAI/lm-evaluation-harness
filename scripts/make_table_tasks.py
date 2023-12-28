"""
Usage:
   Writes csv and Markdown table to csv_file, md_file (below).
"""
import logging
from pathlib import Path

import datasets
import pandas as pd

from lm_eval import tasks
from lm_eval.tasks import TASK_REGISTRY

from ..tests.utils import new_tasks


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
datasets.disable_caching()
tasks.initialize_tasks()


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
        _tasks = tasks.TASK_REGISTRY.items()
        _tasks = sorted(_tasks, key=lambda x: x[0])
    else:
        task_classes = new_tasks()
        _tasks = [(x, TASK_REGISTRY.get(x)) for x in task_classes]
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
