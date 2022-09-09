import argparse
import os
import numpy as np

import lm_eval
from lm_eval.api import utils


EXAMPLE_DIVIDER = "!!@@##@@!! -- Example {i}\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_path", required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--template_names", default="all_templates")
    parser.add_argument("--sets", type=str, default="val")  # example: val,test
    parser.add_argument("--num_fewshot", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=utils.DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    template_names = utils.cli_template_names(args.task_name, args.template_names)
    tasks = lm_eval.get_task_list(args.task_name, template_names)

    os.makedirs(args.output_base_path, exist_ok=True)
    for task, template_name in zip(tasks, template_names):
        iters = []
        for set in args.sets.split(","):
            if set == "train" and task.has_training_docs():
                docs = task.training_docs()
            if set == "val" and task.has_validation_docs():
                docs = task.validation_docs()
            if set == "test" and task.has_test_docs():
                docs = task.test_docs()
            iters.append(docs)
        docs = utils.join_iters(iters)

        file_name = lm_eval.tasks._get_task_template_key(args.task_name, template_name)
        with open(os.path.join(args.output_base_path, file_name), "w") as f:
            for i, doc in (
                zip(range(args.num_examples), docs)
                if args.num_examples > 0
                else enumerate(docs)
            ):
                f.write(EXAMPLE_DIVIDER.format(i=i))
                ctx, _ = task.fewshot_context(
                    doc=doc,
                    num_fewshot=args.num_fewshot,
                    rng=rng,
                )
                f.write(ctx + "\n")


if __name__ == "__main__":
    main()
