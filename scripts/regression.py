import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from lm_eval import utils
from lm_eval.api.registry import ALL_TASKS


seq2seq_models = ["google/flan-t5-small"]
causal_models = [
    "gpt2",
    "facebook/opt-125m",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/pythia-160m",
]
model_names = seq2seq_models + causal_models


completion_tasks = ["boolq", "lambada_openai", "winogrande"]
choice_tasks = ["hellaswag", "openbookqa", "piqa"]
perplexity_tasks = ["wikitext"]
generation_tasks = []
task_names = completion_tasks + choice_tasks + perplexity_tasks + generation_tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", default=[])
    parser.add_argument("--models", default=model_names)
    parser.add_argument("--tasks", default=task_names)
    parser.add_argument("--acc_norm", type=bool, default=False)
    parser.add_argument("--perplexity", default=None)
    # TODO: implement num_fewshot and limit per task, e.g. task1:5,task2:1:100,task3::1000
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None)
    # TODO: implement hf-auto to pick between causal and seq2seq models so we don't need this
    parser.add_argument("--model", default="hf-causal")
    # Use whatever is faster here
    parser.add_argument("--model_args", default="use_accelerate=True,load_in_8bit=True")
    parser.add_argument("--batch_size", default="auto")
    return parser.parse_args()


def eval_models(args, branch=None):
    if branch is not None:
        if os.system(f"git checkout {branch}") != 0:
            return {}, 0

    branch = branch or initial_branch

    start_time = time.time()

    results = {}

    for model in args.models:
        model_type = (
            "hf-causal"
            if model in causal_models
            else "hf-seq2seq"
            if model in seq2seq_models
            else args.model
        )
        model_args = f"pretrained={model},{args.model_args}"
        # TODO: split_and_pad_windows in AutoSeq2SeqLM doesn"t exist, #527
        tasks = (
            args.tasks
            if model in causal_models or model_type == "hf-causal"
            else list(filter(lambda task: task not in perplexity_tasks, args.tasks))
        )
        # TODO: OOM with auto for seq2seq models, also can OOM with llama
        batch_size = (
            args.batch_size
            if model in causal_models or model_type == "hf-causal"
            else 64
            if args.batch_size == "auto"
            else args.batch_size
        )
        output_path = (
            f"data/regression/{int(start_time)}-{branch}-{Path(model).name}.json"
        )

        command = (
            f"python3 main.py --model {model_type} --model_args {model_args} --tasks {','.join(tasks)} "
            f"--num_fewshot {args.num_fewshot}{'' if args.limit is None else f' --limit {args.limit}'} "
            f"--batch_size {batch_size} --no_cache --output_path {output_path}"
        )

        print(
            f"{'=' * 80}\nEvaluating {model} on {', '.join(tasks)} at {branch} with:\n\n{command}\n{'=' * 80}"
        )

        ret = os.system(command)

        results[model] = (
            json.load(open(output_path, encoding="utf-8"))
            if ret == 0
            else {"results": {}}
        )

    end_time = time.time()

    return results, end_time - start_time


def extract_value(args, results, model, task, err=False):
    if model not in results:
        return 0
    results = results[model]["results"]
    if task not in results:
        return 0
    results = results[task]
    if args.acc_norm and "acc_norm,none" in results:
        return results["acc_norm,none"] if not err else results["acc_norm_stderr,none"]
    if "acc,none" in results:
        return results["acc,none"] if not err else results["acc_stderr,none"]
    if (args.perplexity or "word_perplexity") + ",none" in results:
        return (
            results[(args.perplexity or "word_perplexity") + ",none"] if not err else 0
        )
    return 0


def format_value(args, results, model, task):
    val = 100 * extract_value(args, results, model, task)
    err = 100 * extract_value(args, results, model, task, err=True)
    return f"{val:.2f}{f' ± {err:.2f}' if err != 0 else ''}"


def format_diff(args, results1, results2, model, task):
    val1 = 100 * extract_value(args, results1, model, task)
    val2 = 100 * extract_value(args, results2, model, task)
    diff = val2 - val1
    return f"**+{diff:.2f}**" if diff > 0 else f"{diff:.2f}"


def main():
    args = parse_args()

    args.branches = (
        args.branches.split(",") if isinstance(args.branches, str) else args.branches
    )
    args.models = (
        args.models.split(",") if isinstance(args.models, str) else args.models
    )
    args.tasks = (
        ALL_TASKS
        if args.tasks == "all_tasks"
        else utils.pattern_match(args.tasks.split(","), ALL_TASKS)
        if isinstance(args.tasks, str)
        else args.tasks
    )

    global initial_branch
    initial_branch = (
        subprocess.check_output("git branch --show-current", shell=True)
        .decode("ascii")
        .strip()
    )

    # TODO: implement proper timing for each task
    # TODO: reduce IO by sharing tasks between models?

    results, runtime = eval_models(args)
    print(results, runtime)

    runs = []
    for branch in args.branches:
        runs.append((branch, *eval_models(args, branch)))

    os.system(f"git checkout {initial_branch}")

    print("")
    print(f"|task|{'|'.join(map(lambda model: Path(model).name, args.models))}|")
    print(f"|--|{'--|' * len(args.models)}")
    for task in args.tasks:
        print(
            f"|{task} ({initial_branch})|{'|'.join(map(lambda model: format_value(args, results, model, task), args.models))}|"
        )
        for branch, branch_results, branch_runtime in runs:
            print(
                f"|{task} ({branch})|{'|'.join(map(lambda model: format_value(args, branch_results, model, task), args.models))}|"
            )
            print(
                f"|{task} (diff)|{'|'.join(map(lambda model: format_diff(args, results, branch_results, model, task), args.models))}|"
            )

    print("")
    print("|branch|runtime|%|")
    print("|--|--|--|")
    print(f"|{initial_branch}|{runtime:.1f}s|100%|")
    for branch, _, branch_runtime in runs:
        print(f"|{branch}|{branch_runtime:.1f}s|{100 * branch_runtime / runtime:.2f}%|")


if __name__ == "__main__":
    main()
