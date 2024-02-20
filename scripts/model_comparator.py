import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import torch

import lm_eval.evaluator
import lm_eval.models.utils
from lm_eval import tasks, utils


os.environ["TOKENIZERS_PARALLELISM"] = "false"
eval_logger = utils.eval_logger


def memory_stats():
    eval_logger.info(
        f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}, reserved: {torch.cuda.memory_reserved() // 1024 ** 2}"
    )


def calculate_z_value(res1: Dict, res2: Dict) -> Tuple[float, float]:
    acc1, acc2 = res1["acc,none"], res2["acc,none"]
    st_err1, st_err2 = res1["acc_stderr,none"], res2["acc_stderr,none"]
    Z = (acc1 - acc2) / np.sqrt((st_err1**2) + (st_err2**2))
    # Determining the p-value
    p_value = 2 * scipy.stats.norm.sf(abs(Z))  # two-tailed test
    return Z, p_value


def print_results(
    data_to_print: List = None, results_dict: Dict = None, alpha: float = None
):
    model1_data = data_to_print[0]
    model2_data = data_to_print[1]
    table_data = []
    for task in model1_data.keys():
        row = {
            "Task": task,
            "HF Accuracy": model1_data[task]["acc,none"],
            "vLLM Accuracy": model2_data[task]["acc,none"],
            "HF StdErr": model1_data[task]["acc_stderr,none"],
            "vLLM StdErr": model2_data[task]["acc_stderr,none"],
        }
        table_data.append(row)
    comparison_df = pd.DataFrame(table_data)
    comparison_df["Z-Score"] = comparison_df["Task"].apply(
        lambda task: results_dict[task]["z"]
    )
    comparison_df["P-Value"] = comparison_df["Task"].apply(
        lambda task: results_dict[task]["p_value"]
    )
    comparison_df[f"p > {alpha}"] = comparison_df["P-Value"].apply(
        lambda p: "✓" if p > alpha else "×"
    )
    return comparison_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained", default="EleutherAI/pythia-70m", help="name of model to compare"
    )
    parser.add_argument(
        "--hf_args", help="huggingface model args <arg>=<value>", default=""
    )
    parser.add_argument("--vllm_args", help="vllm model args <arg>=<value>", default="")
    parser.add_argument("--tasks", type=str, default="arc_easy,hellaswag")
    parser.add_argument(
        "--limit",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for two-tailed z-test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=8,
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args()


if __name__ == "__main__":
    tasks.initialize_tasks()
    args = parse_args()
    tasks = args.tasks.split(",")
    print(tasks)
    hf_args, vllm_args = "," + args.hf_args, "," + args.vllm_args
    results_vllm = lm_eval.evaluator.simple_evaluate(
        model="vllm",
        model_args=f"pretrained={args.pretrained}" + vllm_args,
        tasks=tasks,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch,
    )
    memory_stats()
    lm_eval.models.utils.clear_torch_cache()
    eval_logger.info("Memory stats cleared")
    memory_stats()
    results_hf = lm_eval.evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.pretrained}" + hf_args,
        tasks=tasks,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch,
    )
    all_res = {}
    for task1, task2 in zip(
        results_hf["results"].items(), results_vllm["results"].items()
    ):
        assert task1[0] == task2[0]
        z, p_value = calculate_z_value(task1[1], task2[1])
        all_res[task1[0]] = {"z": z, "p_value": p_value}
    df = print_results(
        [results_hf["results"], results_vllm["results"]], all_res, args.alpha
    )
    print(df)
