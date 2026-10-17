"""Print a few example prompts from the given tasks.

Usage:
    python scripts/get_prompts.py --tasks arc_easy --num 3
    python scripts/get_prompts.py --tasks my_local_task --include_path my_tasks/
"""

import argparse
from itertools import islice

from lm_eval.tasks import TaskManager


def print_task_prompts(task_name, task, num_examples=3):
    """Print `num_examples` context/target pairs from the task's validation or test split."""
    print("#", task_name)
    print()
    docs = islice(
        task.validation_docs() if task.has_validation_docs() else task.test_docs(),
        num_examples,
    )
    for doc in docs:
        print()
        print("**Context**:", "\n```\n" + task.doc_to_text(doc) + "\n```\n")
        print()
        print("**Target**:", "\n```\n" + task.doc_to_target(doc) + "\n```\n")
        print()


def main(task_names, num_examples=3, task_manager=None):
    """Load the requested tasks and print example prompts for each of them."""
    if task_manager is None:
        task_manager = TaskManager()
    loaded = task_manager.load(task_names)["tasks"]
    for task_name, task in loaded.items():
        print_task_prompts(task_name, task, num_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print a few example prompts from the given tasks."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Task name(s) to print prompts for, as indexed by TaskManager.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="Number of example prompts to print per task.",
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
    main(args.tasks, num_examples=args.num, task_manager=task_manager)
