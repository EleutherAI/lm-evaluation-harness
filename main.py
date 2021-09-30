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
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no_cache', action="store_true")
    return parser.parse_args()

def main():

    args = parse_args()

    assert not args.provide_description # not implemented
    
    if args.limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")

    results = evaluator.simple_evaluate(args.model, args.model_args, task_names, args.num_fewshot, args.batch_size, args.device, args.no_cache, args.limit)

    dumped = json.dumps(results, indent=2)
    
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}")
    print(evaluator.make_table(results))

if __name__ == "__main__":
    main()
