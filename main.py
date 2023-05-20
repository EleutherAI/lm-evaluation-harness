import argparse
import json
import logging
import fnmatch
import yaml
import os

from lm_eval import evaluator, tasks
from lm_eval.api.task import ConfigurableTask, TASK_REGISTRY

logging.getLogger("openai").setLevel(logging.WARNING)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ALL_TASKS = sorted(list(TASK_REGISTRY))


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices
        print(f"{ALL_TASKS} is this")

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(ALL_TASKS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


def main():
    args = parse_args()

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        if args.config:
            task_names = []
            for config_files in args.config.split(","):
                with open(config_files, "r") as f:
                    config = yaml.load(f, yaml.Loader)

                if args.num_fewshot != 0:
                    config["num_fewshot"] = args.num_fewshot

                if args.batch_size != None:
                    config["batch_size"] = args.batch_size

                task_names.append(config)
        else:
            task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
    )
    if results is not None:
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
