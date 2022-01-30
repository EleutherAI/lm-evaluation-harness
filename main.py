import argparse
import json
import logging
import fnmatch

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)

class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice

# Get task base classes for filtering
task_types = list(set([task.__bases__[0].__name__ for task in tasks.TASK_REGISTRY.values()]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument('--task_type', default=None, choices=MultiChoice(task_types))    
    parser.add_argument('--exclude_tasks', default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no_cache', action="store_true")
    parser.add_argument('--decontaminate', action="store_true")
    parser.add_argument('--ngrams_path', default=None)
    parser.add_argument('--ngrams_n_size', type=int, default=None)

    return parser.parse_args()

def ensure_correct_decontamination_params(args):
    valid = True
    if args.decontaminate:
        if not args.ngrams_n_size:
            print("Please specify n size of training set n-grams. (--ngrams_n_size)")
            valid = False
        if not args.ngrams_path:
            print("Please specify path containing training set n-grams. (--ngrams_path)")
            valid = False

    return valid

# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def main():
    parser.add_argument('--description_dict_path', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not ensure_correct_decontamination_params(args):
        return
        
    assert not args.provide_description  # not implemented
    
    if args.limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.task_type:
        task_types = args.task_type.split(",")
        task_names = list(dict(filter(lambda x: x[1].__bases__[0].__name__ in task_types,
                                      tasks.TASK_REGISTRY.items())
                                      ).keys())

    if args.tasks is None:
        if args.task_type is None:
            task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    if args.exclude_tasks:
        exclude_tasks = pattern_match(args.exclude_tasks.split(","), task_names)
        task_names = list(filter(lambda x: x not in exclude_tasks, task_names))

    if len(task_names) == 0:
        print("You must have excluded the tasks you specified, exiting.")
        return

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, 'r') as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontaminate=args.decontaminate,
        ngrams_path=args.ngrams_path,
        ngrams_n_size=args.ngrams_n_size
    )

    dumped = json.dumps(results, indent=2)    
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
