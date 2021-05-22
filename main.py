import argparse
import json
import numpy as np
import random
import logging

from lm_eval import models, tasks, evaluator, base

logging.getLogger("openai").setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no_cache', action="store_true")
    return parser.parse_args()

def main():

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    lm = models.get_model(args.model).create_from_arg_string(args.model_args, {
        'batch_size': args.batch_size, 'device': args.device
    })
    
    if args.limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if not args.no_cache:
        lm = base.CachingLM(lm, 'lm_cache/' + args.model + '_' + args.model_args.replace('=', '-').replace(',', '_').replace('/', '-') + '.db')
    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)

    results = evaluator.evaluate(lm, task_dict, args.provide_description, args.num_fewshot, args.limit)

    dumped = json.dumps(results, indent=2)
    print(dumped)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    # MAKE TABLE
    from pytablewriter import MarkdownTableWriter

    writer = MarkdownTableWriter()
    writer.headers = ["Task", "Metric", "Value"]

    values = []

    for k, dic in results.items():
        for m, v in dic.items():
            values.append([k, m, '%.4f' % v])
            k = ""
    writer.value_matrix = values

    print(writer.dumps())

if __name__ == "__main__":
    main()
