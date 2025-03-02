import argparse

from lm_eval import tasks
from lm_eval import utils
from lm_eval.evaluator_utils import get_task_list
from lm_eval.tasks import TaskManager
from lm_eval.utils import eval_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--local_base_dir")
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if args.tasks == "all_tasks":
        task_names = task_manager.all_tasks
    else:
        task_list = args.tasks.split(",")

        task_names = task_manager.match_tasks(task_list)

        task_missing = [
            task for task in task_list if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used

        if task_missing:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}\n"
                f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
            )
            raise ValueError(
                f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
            )

    task_dict = tasks.get_task_dict(task_names, task_manager)

    for task in [x.task for x in get_task_list(task_dict)]:
        task.save_to_disk(args.local_base_dir)

if __name__ == "__main__":
    main()
