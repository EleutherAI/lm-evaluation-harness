import os
import re
import sys
import types

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
    task_name: list[str], limit: int, model: str, model_args: str, bootstrap_iters: int
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
    task_dict = task_manager.load(task_name)

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
        for x, y in zip(
            [y for _, y in r(e1).items()], [y for _, y in r(e2).items()], strict=True
        )
    )


@pytest.mark.parametrize(
    "task_name,limit,model,model_args",
    [
        (
            ["ai2_arc"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["mmlu_stem"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["lambada_openai"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["wikitext"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
    ],
    ids=lambda d: f"{d}",
)
def test_printed_results(
    task_name: list[str], limit: int, model: str, model_args: str, on_ci: bool
):
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
            re.sub(r"[^a-zA-Z0-9_\-.]", "-", model_args),
        )
    )
    filepath = f"./tests/testdata/{filename}.txt"
    with open(filepath) as f:
        t1 = f.read().strip()

    t2 = make_table(results).strip()

    t1_lines, t2_lines = t1.splitlines(), t2.splitlines()
    assert len(t1_lines) == len(t2_lines)
    for t1_line, t2_line in zip(t1_lines, t2_lines, strict=True):
        t1_items, t2_items = t1_line.split("|"), t2_line.split("|")
        assert len(t1_items) == len(t2_items)
        for t1_item, t2_item in zip(t1_items, t2_items, strict=True):
            try:
                t1_item_f = float(t1_item)
                t2_item_f = float(t2_item)
                ## TODO: these are pretty loose tolerances but:
                # - we only test 10 samples
                # - not sure when/how the ground truth test_data was generated
                tol = 0.3 if on_ci else 0.5
                assert abs(t1_item_f - t2_item_f) < tol
            except ValueError:
                # Strip whitespace so column-width differences
                # (caused by value precision changes) don't fail the test.
                # Also ignore separator-line cells (e.g. "------:").
                t1_s = t1_item.strip().rstrip("-:").rstrip("-")
                t2_s = t2_item.strip().rstrip("-:").rstrip("-")
                if t1_s or t2_s:
                    assert t1_s == t2_s
                #     assert t1_s == t2_s


class _FakeMarkdownTableWriter:
    instances = []

    def __init__(self):
        self.headers = []
        self.value_matrix = []
        type(self).instances.append(self)

    def dumps(self):
        return repr(self.value_matrix)


class _FakeLatexTableWriter(_FakeMarkdownTableWriter):
    pass


@pytest.fixture
def fake_pytablewriter(monkeypatch):
    _FakeMarkdownTableWriter.instances.clear()
    _FakeLatexTableWriter.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "pytablewriter",
        types.SimpleNamespace(
            MarkdownTableWriter=_FakeMarkdownTableWriter,
            LatexTableWriter=_FakeLatexTableWriter,
        ),
    )


def test_make_table_regression_preserves_hierarchy_and_metadata(fake_pytablewriter):
    result_dict = {
        "results": {
            "parent": {
                "alias": "Parent",
                "name": "ignore-me",
                "sample_len": 123,
                "sample_count": {"acc,none": 123},
                "acc,none": 0.9,
                "acc_stderr,none": 0.01,
            },
            "child": {
                "alias": "Child",
                "sample_len": 50,
                "acc,none": 0.8,
            },
            "standalone": {
                "alias": "Standalone",
                "loss,none": 0.2,
            },
        },
        "versions": {"parent": 1, "child": 2, "standalone": 3},
        "n-shot": {"parent": 5, "child": 0, "standalone": 2},
        "higher_is_better": {
            "parent": {"acc": True},
            "child": {"acc": True},
            "standalone": {"loss": False},
        },
        "group_subtasks": {"parent": ["child"]},
    }

    output = make_table(result_dict)

    assert output == repr(
        [
            ["Parent", 1, "none", "5", "acc", "↑", "0.9000", "±", "0.0100"],
            [" - Child", 2, "none", "0", "acc", "↑", "0.8000", "", ""],
            ["Standalone", 3, "none", "2", "loss", "↓", "0.2000", "", ""],
        ]
    )
    assert _FakeMarkdownTableWriter.instances[0].headers == [
        "Tasks",
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]


def test_make_table_regression_sorted_results_is_alphabetical(fake_pytablewriter):
    result_dict = {
        "results": {
            "parent": {"alias": "Parent", "acc,none": 0.9},
            "child": {"alias": "Child", "acc,none": 0.8},
            "standalone": {"alias": "Standalone", "acc,none": 0.7},
        },
        "versions": {},
        "group_subtasks": {"parent": ["child"]},
    }

    make_table(result_dict, sort_results=True)

    assert _FakeMarkdownTableWriter.instances[0].value_matrix == [
        [" - Child", "    N/A", "none", " ", "acc", "", "0.8000", "", ""],
        ["Parent", "    N/A", "none", " ", "acc", "", "0.9000", "", ""],
        ["Standalone", "    N/A", "none", " ", "acc", "", "0.7000", "", ""],
    ]
