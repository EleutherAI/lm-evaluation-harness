"""Print a table of task split and metric information.

Usage:
    python scripts/make_table_tasks.py --tasks arc_easy --output task_table.md
"""

import argparse
import logging

from pytablewriter import MarkdownTableWriter

from lm_eval.tasks import TaskManager


logger = logging.getLogger(__name__)


def check(tf):
    if tf:
        return "✓"
    else:
        return " "


def make_table(task_manager, task_names):
    """Generate a markdown table describing the requested tasks."""
    writer = MarkdownTableWriter()
    writer.headers = ["Task Name", "Train", "Val", "Test", "Val/Test Docs", "Metrics"]

    values = []

    for name, task in task_manager.load(task_names)["tasks"].items():
        docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
        values.append(
            [
                name,
                check(task.has_training_docs()),
                check(task.has_validation_docs()),
                check(task.has_test_docs()),
                len(list(docs)),
                ", ".join(task.aggregation().keys()),
            ]
        )
    writer.value_matrix = values
    return writer.dumps()


def main(task_names, output="task_table.md", task_manager=None):
    """Write the generated table to `output` and return it."""
    if task_manager is None:
        task_manager = TaskManager()

    table = make_table(task_manager, task_names)
    with open(output, "w", encoding="utf-8") as f:
        f.write(table)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print split and metric information for the given tasks."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Task name(s) to describe, as indexed by TaskManager.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="task_table.md",
        help="Path of the markdown file to write.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to scan for task configs (forwarded to TaskManager).",
    )
    args = parser.parse_args()
    task_manager = (
        TaskManager(include_path=args.include_path)
        if args.include_path
        else TaskManager()
    )
    main(args.tasks, args.output, task_manager)
