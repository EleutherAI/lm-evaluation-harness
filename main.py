import argparse
import json
import numpy as np
import random
import itertools
import collections

from lm_eval import models, tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--limit', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    lm = models.get_model(args.model).create_from_arg_string(args.model_args)
    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)
    task_dict_items = list(task_dict.items())
    results = {}

    requests = collections.defaultdict(list)
    requests_lengths = collections.defaultdict(list)

    for task_name, task in task_dict_items:
        # TODO: fall back to test docs
        if not task.has_validation_docs():
            continue

        for doc in itertools.islice(task.validation_docs(), 0, args.limit):
            ctx = task.fewshot_context(
                doc=doc,
                provide_description=args.provide_description,
                num_fewshot=args.num_fewshot,
            )

            reqs = task.construct_requests(ctx)

            lengths = collections.defaultdict(int)

            for req in reqs:
                requests[req.type].append(req)
                lengths[req.type] += 1
            
            for type, ct in lengths.items():
                requests_lengths[type].append(ct)

    # TODO: finish implementation
    for reqname, reqs in requests.items():
        lm_res = getattr(lm, reqname)([req.args for req in reqs])

    for task_name, task in task_dict_items:
        if not task.has_validation_docs():
            continue


    dumped = json.dumps(results, indent=2)
    print(dumped)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
