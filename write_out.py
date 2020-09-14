import argparse
import numpy as np
import os
import random

from lm_eval import tasks

EXAMPLE_DIVIDER = "!!@@##@@!! -- Example {i}\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_base_path', required=True)
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_examples', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)
    os.makedirs(args.output_base_path, exist_ok=True)
    for task_name, task in task_dict.items():
        if not task.has_validation_docs():
            continue
        docs = task.validation_docs()
        with open(os.path.join(args.output_base_path, task_name), "w") as f:
            for i, doc in zip(range(args.num_examples), docs):
                f.write(EXAMPLE_DIVIDER.format(i=i))
                ctx = task.fewshot_context(
                    doc=doc,
                    provide_description=args.provide_description,
                    num_fewshot=args.num_fewshot,
                )
                f.write(ctx + "\n")


if __name__ == "__main__":
    main()
