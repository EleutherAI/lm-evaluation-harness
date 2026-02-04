import argparse
import logging
import os
import random
from typing import cast

import numpy as np

from lm_eval.tasks import TaskManager
from lm_eval.utils import join_iters


eval_logger = logging.getLogger(__name__)


EXAMPLE_DIVIDER = "!!@@##@@!! -- Example {i}\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_path", "--output_path", required=True)
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--sets", type=str, default="val")  # example: val,test
    parser.add_argument("--num_fewshot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_examples", type=int, default=1)
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
    np.random.seed(args.seed)

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    task_manager = TaskManager(include_path=args.include_path)

    if args.tasks == "all_tasks":
        _task_names = task_manager.all_tasks
    else:
        _task_names = cast("list[str]", args.tasks.split(","))
    _res = task_manager.load(_task_names)
    task_dicts = _res["tasks"].values()

    os.makedirs(args.output_base_path, exist_ok=True)
    for task in task_dicts:
        task_name = task.config.task
        rnd = random.Random()
        rnd.seed(args.seed)

        iters = []

        for set in args.sets.split(","):
            docs = None
            if set == "train" and task.has_training_docs():
                docs = task.training_docs()
            if set == "val" and task.has_validation_docs():
                docs = task.validation_docs()
            if set == "test" and task.has_test_docs():
                docs = task.test_docs()
            if docs is not None:
                iters.append(docs)

        if len(iters) == 0:
            raise ValueError(
                f"Passed --sets '{args.sets}' but this task has no splits which match. Please specify a different --sets value."
            )

        docs = join_iters(iters)

        with open(
            os.path.join(args.output_base_path, task_name),  # type: ignore
            "w",
            encoding="utf8",
        ) as f:
            for i, doc in (
                zip(range(args.num_examples), docs, strict=False)
                if args.num_examples > 0
                else enumerate(docs)
            ):
                f.write(EXAMPLE_DIVIDER.format(i=i))
                ctx = task.fewshot_context(
                    doc=doc,
                    num_fewshot=args.num_fewshot,
                )
                f.write(ctx + "\n")


if __name__ == "__main__":
    main()
