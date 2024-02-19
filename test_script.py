import os

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

# Used to specify alternate cache path, useful if run in a docker container
# NOTE raw datasets will break if you try to transfer the cache from your host to an image
LM_HARNESS_CACHE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")

import torch
from transformers import (
    pipeline as trans_pipeline,
)
import torch

from lm_eval import simple_evaluate, tasks as tasks_class

device = "cuda" if torch.cuda.is_available() else "cpu"

model = "gpt2"

local_models_path = os.getenv("MODELS_PATH")

task = "text-generation"

print("\n\n\n")

print("Model is: ", model)

print("This task is: ", task)


def run_model(model: str, device: str):
    trans_pipe = trans_pipeline(
        task=task, model=model, device=device, trust_remote_code=True
    )

    model = trans_pipe.model
    tokenizer = trans_pipe.tokenizer

    tasks = [
        # "squadv2",
        # "swag",
        "hellaswag",
        "lambada_openai",
        # "glue",
        # "super-glue-lm-eval-v1",
        # "wikitext",
        # "winogrande"
    ]

    eval_data = simple_evaluate(
        model="hf-auto",
        model_args={
            "pretrained": model,
            "tokenizer": tokenizer,
        },
        limit=0.01,
        device=device,
        cache_requests=True,
        # rewrite_requests_cache=True,
        # delete_requests_cache=True,
        tasks=tasks,
        write_out=True,
    )

    return eval_data


if __name__ == "__main__":
    eval_data = run_model(model=model, device=device)

    results: dict = eval_data["results"]

    for dataset, metrics_dict in results.items():
        for key, value in metrics_dict.items():
            if key == "acc,none":
                print(f"Dataset: {dataset}, Accuracy: {value}")


pass
