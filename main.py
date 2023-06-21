import os
import re
import json
import fnmatch
import jsonlines
import argparse
import logging

from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.logger import eval_logger
from lm_eval.tasks import include_task_folder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(sorted(ALL_TASKS))
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--include_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--use_cache", type=str, default=None)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--log_samples", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_task_folder(args.include_path)

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = args.tasks.split(",")
            task_names = utils.pattern_match(tasks_list, ALL_TASKS)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=lambda o: str(o))
        print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.output_path:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

            with open(args.output_path, "w") as f:
                f.write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(
                        re.sub("/", "__", args.model_args), task_name
                    )
                    if os.path.isdir(args.output_path):
                        filename = f"./{args.output_path}/{output_name}.jsonl"
                    elif os.path.isfile(args.output_path):
                        filename = (
                            f"./{os.path.dirname(args.output_path)}/{output_name}.jsonl"
                        )

                    with jsonlines.open(filename, "w") as f:
                        f.write_all(samples[task_name])

        print(
            f"{args.model} ({args.model_args}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
