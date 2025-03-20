import os
import re
from typing import List

import pytest

import lm_eval.api as api
import lm_eval.evaluator as evaluator
from lm_eval import tasks
from lm_eval.utils import make_table


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces


@pytest.mark.parametrize(
    "task_name,limit,model,model_args,bootstrap_iters",
    [
        (
            ["arc_easy"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
            0,
        ),
        (
            ["mmlu_abstract_algebra"],
            None,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
            10000,
        ),
    ],
    ids=lambda d: f"{d}",
)
def test_evaluator(
    task_name: List[str], limit: int, model: str, model_args: str, bootstrap_iters: int
):
    e1 = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
        bootstrap_iters=bootstrap_iters,
    )
    assert e1 is not None

    lm = api.registry.get_model(model).create_from_arg_string(
        model_args,
        {
            "batch_size": None,
            "max_batch_size": None,
            "device": None,
        },
    )
    task_manager = tasks.TaskManager()
    task_dict = tasks.get_task_dict(task_name, task_manager)

    e2 = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
    )

    assert e2 is not None
    # check that caching is working

    def r(x):
        if "arc_easy" in x["results"]:
            return x["results"]["arc_easy"]
        else:
            return x["results"]["mmlu_abstract_algebra"]

    assert all(
        x == y
        for x, y in zip([y for _, y in r(e1).items()], [y for _, y in r(e2).items()])
    )


@pytest.mark.parametrize(
    "task_name,limit,model,model_args",
    [
        (
            ["ai2_arc"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m,dtype=float32,device=cpu",
        ),
        (
            ["mmlu_stem"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m,dtype=float32,device=cpu",
        ),
        (
            ["lambada_openai"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m,dtype=float32,device=cpu",
        ),
        (
            ["wikitext"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m,dtype=float32,device=cpu",
        ),
    ],
    ids=lambda d: f"{d}",
)
def test_printed_results(task_name: List[str], limit: int, model: str, model_args: str):
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
        bootstrap_iters=0,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )

    filename = "_".join(
        (
            "-".join(task_name),
            str(limit),
            str(model),
            re.sub(r"[^a-zA-Z0-9_\-\.]", "-", model_args),
        )
    )
    filepath = f"./tests/testdata/{filename}.txt"
    with open(filepath, "r") as f:
        t1 = f.read().strip()

    t2 = make_table(results).strip()

    t1_lines, t2_lines = t1.splitlines(), t2.splitlines()
    assert len(t1_lines) == len(t2_lines)
    for t1_line, t2_line in zip(t1_lines, t2_lines):
        t1_items, t2_items = t1_line.split("|"), t2_line.split("|")
        assert len(t1_items) == len(t2_items)
        for t1_item, t2_item in zip(t1_items, t2_items):
            try:
                t1_item = float(t1_item)
                t2_item = float(t2_item)
                assert abs(t1_item - t2_item) < 0.3
            except ValueError:
                assert t1_item == t2_item
