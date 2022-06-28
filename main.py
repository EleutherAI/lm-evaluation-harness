import argparse
import datetime
import json
import logging
import os
from codecarbon import OfflineEmissionsTracker

from lm_eval import tasks, evaluator


logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
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
    return parser.parse_args()


def args_to_name(args):
    """Map `args` to file name. If output_path is set, we use that instead."""
    if args.output_path is not None:
        return args.output_path

    def _fix_model_name(model, model_args):
        if model_args == "":
            return model
        elif "pretrained" not in model_args:
            logger.warning("WARNING: Unprepared for these model args.")
            return f"{model}_{model_args}"

        for arg in model_args.split(","):
            # Example:
            #   pretrained=google/t5-base-lm-adapt --> google-t5-base-lm-adapt
            if "pretrained" in arg:
                return arg.split("=")[-1].replace("/", "-")

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
    example_logger = logging.getLogger("examples")
    filename = f"./outputs/examples-{output_path}.jsonl"
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    example_logger.addHandler(handler)
    example_logger.setLevel(logging.INFO)


def main():
    os.makedirs("./outputs", exist_ok=True)
    args = parse_args()

    if args.limit:
        logger.warning(
            "\n» WARNING: `--limit` SHOULD ONLY BE USED FOR TESTING. REAL METRICS "
            "SHOULD NOT BE COMPUTED USING LIMIT."
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

    print()  # Ensure a newline after `main` command.
    with OfflineEmissionsTracker(country_iso_code="FRA", log_level="error"):
        print()  # Add newline between emissions tracker and evaluation logging.
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
            seed=args.seed,
        )

    with open(f"./outputs/agg-{output_path}.json", "w") as f:
        json.dump({"results": results["results"], "config": results["config"]}, f)

    from scripts.agg2slim import agg2slim

    with open(f"./outputs/slim-{output_path}.json", "w") as f:
        json.dump(agg2slim(results), f, indent=2)

    print(f"\n{evaluator.make_table(results)}")
    emissions_output_path = f"./outputs/emissions-{output_path}.csv"
    os.rename("emissions.csv", emissions_output_path)


if __name__ == "__main__":
    main()
