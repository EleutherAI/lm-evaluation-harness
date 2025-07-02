"""
Usage:
   python requests_caching.py --tasks=comma,separated,list,of,tasks --cache_requests=<true|refresh|delete]>
"""

import argparse
import logging
import os
from typing import List

import torch
from transformers import (
    pipeline as trans_pipeline,
)

from lm_eval import simple_evaluate
from lm_eval.evaluator import request_caching_arg_to_dict


eval_logger = logging.getLogger(__name__)


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

# Used to specify alternate cache path, useful if run in a docker container
# NOTE raw datasets will break if you try to transfer the cache from your host to a docker image
LM_HARNESS_CACHE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "EleutherAI/pythia-70m"

TASK = "text-generation"


def run_model_for_task_caching(tasks: List[str], cache_requests: str):
    eval_logger.info(f"Loading HF model: {MODEL}")

    trans_pipe = trans_pipeline(
        task=TASK, model=MODEL, device=DEVICE, trust_remote_code=True
    )

    model = trans_pipe.model
    tokenizer = trans_pipe.tokenizer

    eval_logger.info(
        f"Running simple_evaluate to cache request objects for tasks: {tasks}"
    )

    cache_args = request_caching_arg_to_dict(cache_requests=cache_requests)

    eval_logger.info(
        f"The following operations will be performed on the cache: {cache_requests}"
    )

    eval_data = simple_evaluate(
        model="hf-auto",
        model_args={
            "pretrained": model,
            "tokenizer": tokenizer,
        },
        limit=1,
        device=DEVICE,
        tasks=tasks,
        write_out=True,
        **cache_args,
    )

    return eval_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        metavar="task1,task2",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )

    args = parser.parse_args()

    tasks = args.tasks.split(",")

    eval_data = run_model_for_task_caching(
        tasks=tasks, model=MODEL, device=DEVICE, cache_requests=args.cache_requests
    )
