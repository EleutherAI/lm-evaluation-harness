import argparse
import numpy as np
import json
import os
import random
from lm_eval import tasks
from lm_eval.utils import join_iters

EXAMPLE_DIVIDER = "!!@@##@@!! -- Example {i}\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_path", required=True)
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--sets", type=str, default="val")  # example: val,test
    parser.add_argument("--num_fewshot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--description_dict_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    os.makedirs(args.output_base_path, exist_ok=True)
    for task_name, task in task_dict.items():
        rnd = random.Random()
        rnd.seed(args.seed)

        iters = []

        for set in args.sets.split(","):
            if set == "train" and task.has_training_docs():
                docs = task.training_docs()
            if set == "val" and task.has_validation_docs():
                docs = task.validation_docs()
            if set == "test" and task.has_test_docs():
                docs = task.test_docs()
            iters.append(docs)

        docs = join_iters(iters)

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )

        with open(os.path.join(args.output_base_path, task_name), "w") as f:
            for i, doc in (
                zip(range(args.num_examples), docs)
                if args.num_examples > 0
                else enumerate(docs)
            ):
                f.write(EXAMPLE_DIVIDER.format(i=i))
                ctx = task.fewshot_context(
                    doc=doc,
                    num_fewshot=args.num_fewshot,
                    rnd=rnd,
                    description=description,
                )
                f.write(ctx + "\n")


if __name__ == "__main__":
    main()
