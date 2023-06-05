import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--data_dirs", default="",
            help="comma-delimited list of task_name=data_dir arguments used for loading task"
                "datasets with datasets.load_dataset()")
    parser.add_argument("--cache_dirs", default="",
            help="comma-delimited list of task_name=cache_dir arguments used for loading task"
                "datasets with datasets.load_dataset()")
    parser.add_argument("--download_modes", default="",
            help="comma-delimited list of task_name=download_mode arguments used for loading"
                "task datasets with datasets.load_dataset()")

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    # Get named arguments to Task constructor, which are used for setting data_dir, cache_dir, and
    # download_mode for the Task constructor.
    data_dirs = utils.extract_args(args.data_dirs.split(","))
    cache_dirs = utils.extract_args(args.cache_dirs.split(","))
    download_modes = utils.extract_args(args.download_modes.split(","))
    task_init_args = {k: dict() for k in data_dirs.keys() | cache_dirs.keys() | download_modes.keys()}
    for k in task_init_args.keys():
        if k in data_dirs:
            task_init_args[k] = {**task_init_args[k], **{"data_dir": data_dirs[k]}}
        if k in cache_dirs:
            task_init_args[k] = {**task_init_args[k], **{"cache_dir": cache_dirs[k]}}
        if k in download_modes:
            task_init_args[k] = {**task_init_args[k], **{"download_mode": download_modes[k]}}
    print(f"Data directories: {data_dirs}")
    print(f"Cache directories: {cache_dirs}")
    print(f"Download modes: {download_modes}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
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
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        task_init_args=task_init_args,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
