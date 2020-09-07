import argparse
import json

from lm_eval import models, tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    lm = models.get_model(args.model).create_from_arg_string(args.model_args)
    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = {
        task_name: tasks.get_task(task_name)()
        for task_name in task_names
    }
    results = {}
    for task_name, task in task_dict.items():
        if not task.has_validation_docs():
            continue
        result = task.evaluate(
            docs=task.validation_docs(),
            lm=lm,
            provide_description=args.provide_description,
            num_fewshot=args.num_fewshot,
        )
        results[task_name] = result
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
