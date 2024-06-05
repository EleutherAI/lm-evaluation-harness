import argparse
import os
import random
import json

import numpy as np

from lm_eval import tasks
from lm_eval.tasks import TaskManager
from lm_eval.utils import eval_logger, join_iters

# python -m scripts.write_out_json --output_base_path /data/richard_ren/lm-evaluation-harness/data/TASKNAME --num_fewshot 0 --tasks TASK_NAME

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_path", "--output_path", required=True)
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--sets", type=str, default="test")  # example: val,test
    parser.add_argument("--num_fewshot", type=int, default=0) # modify this script to check if the fewshot is in the yaml
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_examples", type=int, default=0)
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

    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if args.tasks == "all_tasks":
        task_names = task_manager.all_tasks
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names, task_manager)

    print(task_dict, task_dict.items())

    os.makedirs(args.output_base_path, exist_ok=True)
    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            _, task = task
            if task is None:
                continue
        rnd = random.Random()
        rnd.seed(args.seed)

        iters = []
        
        print(task_name, task)
        for set in args.sets.split(","):
            docs = None
            if task.has_training_docs():
                print("has training docs")
                docs = task.training_docs()
                print(docs)
            if task.has_validation_docs():
                print("has validation docs")
                docs = task.validation_docs()
                print(docs)
            if task.has_test_docs:
                print("has test docs")
                docs = task.test_docs()
                print(docs)
            if set == "train" and task.has_training_docs():
                docs = task.training_docs()
                print("selected train")
            if set == "val" and task.has_validation_docs():
                docs = task.validation_docs()
                print("selected val")
            if set == "test" and task.has_test_docs():
                docs = task.test_docs()
                print("selected test")
            if docs is not None:
                iters.append(docs)

        docs = join_iters(iters)

        with open(
            os.path.join(args.output_base_path, f"{task_name}.jsonl"), "w", encoding="utf8"
        ) as f:
            for i, doc in (
                zip(range(args.num_examples), docs)
                if args.num_examples > 0
                else enumerate(docs)
            ):
                # print(f"Writing example {i} for task {task_name}")
                # print(task)
                # print(doc)
                # print()
                ctx = task.fewshot_context(
                    doc=doc,
                    num_fewshot=args.num_fewshot,
                )
                
               # ['Find hash collisions', 'Compute injection payloads', 'Deserialize and serialize json objects', 'Serialize java payloads']
                number = task.doc_to_target(
                    doc=doc,
                )
                # 1
                construct_requests = task.construct_requests(
                    doc=doc, ctx=ctx
                )

                labels = {}
                try:
                    choices = task.doc_to_choice(
                        doc=doc
                    ) 
                    for idx, choice in enumerate(choices):
                        labels[choice] = 1 if (number == idx) or (number == choice) else 0
                except:
                    #print("not multiple choice")
                    pass
                # print(options)
                # print("%%%%%")
                # print(construct_requests)
                example = {
                    "prompts": ctx,
                    "labels": labels
                }
                f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
