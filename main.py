import argparse
import datetime
import json
import logging
import os

from lm_eval import tasks, evaluator
from codecarbon import OfflineEmissionsTracker

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--parallelize", action="store_true")

    parser.add_argument(
        "--output_path",
        default=None,
        help="""Use output_path as `output_filename`. For example:
    Currently, you cannot change/add folder structure.

    > python main.py ... --output_path blop
    # saves files into `outputs/blop.json`.
    """,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    return parser.parse_args()


def args_to_name(args):
    """Map `args` to file name. If output_path is set, we use that instead."""
    if args.output_path is not None:
        return args.output_path

    def _fix_model_name(model, model_args):
        if model_args == "":
            return model
        elif "pretrained" in model_args:
            # pretrained=google/t5-base-lm-adapt --> google-t5-base-lm-adapt
            return model_args.split("=")[-1].replace("/", "-")
        else:
            print("WARNING: Unprepared for these model args.")
            return f"{model}_{model_args}"

    fields = [
        _fix_model_name(args.model, args.model_args),
        args.tasks,
        str(args.num_fewshot),
        str(args.seed),
        datetime.datetime.now().isoformat(),
    ]
    fields = [f for f in fields if f is not None]
    # Some prompts also have "/" in them!
    filename = "_".join(fields).replace("/", "-")
    if args.limit is not None:
        # Do not use limited files for final analysis.
        return f"limited_{args.limit}_" + filename

    return filename


def setup_example_logger(output_path):
    """Sets up a logger that will save each example and prediction."""
    logger = logging.getLogger("examples")
    filename = f"./outputs/examples-{output_path}.jsonl"
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main():
    os.makedirs("./outputs", exist_ok=True)
    args = parse_args()
    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    output_path = args_to_name(args)
    setup_example_logger(output_path)

    with OfflineEmissionsTracker(country_iso_code="FRA", log_level="error"):
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
            check_integrity=args.check_integrity,
            seed=args.seed,
            parallelize=args.parallelize,
        )

    with open(f"./outputs/agg-{output_path}.json", "w") as f:
        json.dump({"results": results["results"], "config": results["config"]}, f)

    from scripts.agg2slim import agg2slim

    with open(f"./outputs/slim-{output_path}.json", "w") as f:
        # We add `indent = 2` to help with quick readability.
        json.dump(
            agg2slim(results),
            f,
            indent=2,
        )
    print(evaluator.make_table(results))
    emissions_output_path = f"./outputs/emissions-{output_path}.csv"
    os.rename("emissions.csv", emissions_output_path)


if __name__ == "__main__":
    main()
