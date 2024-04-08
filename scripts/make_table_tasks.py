"""
Usage:
   python make_table_tasks.py --output <markdown_filename>
"""
import argparse
import logging

from pytablewriter import MarkdownTableWriter

from lm_eval import tasks


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check(tf):
    if tf:
        return "✓"
    else:
        return " "


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="task_table.md")
    args = parser.parse_args()

    writer = MarkdownTableWriter()
    writer.headers = ["Task Name", "Train", "Val", "Test", "Val/Test Docs", "Metrics"]
    values = []

    tasks = tasks.TASK_REGISTRY.items()
    tasks = sorted(tasks, key=lambda x: x[0])
    for tname, Task in tasks:
        task = Task()
        v = [
            tname,
            check(task.has_training_docs()),
            check(task.has_validation_docs()),
            check(task.has_test_docs()),
            len(
                list(
                    task.test_docs() if task.has_test_docs() else task.validation_docs()
                )
            ),
            ", ".join(task.aggregation().keys()),
        ]
        logger.info(v)
        values.append(v)
    writer.value_matrix = values
    table = writer.dumps()
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(table)
