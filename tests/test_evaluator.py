import json
import os
from pathlib import Path

import datasets
import pytest

from lm_eval import api, evaluator, tasks
from lm_eval.utils import make_table


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces

MODEL_REVISION = "7386d9a4ae45aef494a6e704910394def3037fc5"
MODEL_ARGS = (
    f"pretrained=EleutherAI/pythia-14m-deduped,revision={MODEL_REVISION},dtype=float32"
)
DATASET_REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "EleutherAI/lambada_openai": "900124bf3b8235c6daf21033af9948b3f07346c4",
    "EleutherAI/wikitext_document_level": ("647234772b9554e208af6c826f23b99e3cac88c8"),
}


def _pin_dataset_revisions(monkeypatch):
    """Pin every dataset loaded by these regression tests to an exact commit."""
    load_dataset = datasets.load_dataset
    observed = {}

    def load_pinned_dataset(path, name=None, **kwargs):
        if path not in DATASET_REVISIONS:
            raise AssertionError(f"missing pinned dataset revision for {path!r}")
        revision = DATASET_REVISIONS[path]
        kwargs["revision"] = revision
        observed[(path, name)] = revision
        return load_dataset(path=path, name=name, **kwargs)

    monkeypatch.setattr(datasets, "load_dataset", load_pinned_dataset)
    return observed


def _evaluate_cpu(task_names, *, task_manager=None):
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=MODEL_ARGS,
        tasks=task_names,
        num_fewshot=0,
        batch_size="1",
        device="cpu",
        limit=10,
        bootstrap_iters=0,
        task_manager=task_manager,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    assert results is not None
    assert results["config"]["model_revision"] == MODEL_REVISION
    assert results["config"]["model_sha"] == MODEL_REVISION
    return results


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
        x == y for x, y in zip(list(r(e1).values()), list(r(e2).values()), strict=True)
    )


def _load_regression_snapshot(name: str):
    path = Path(__file__).parent / "testdata" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _table_structure(table: str):
    """Ignore numeric score drift while retaining table labels and hierarchy."""
    rows = []
    for row_index, line in enumerate(table.splitlines()):
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if cells and all(cell and set(cell) <= {"-", ":"} for cell in cells):
            cells = [
                f"{':' if cell.startswith(':') else ''}"
                f"-{':' if cell.endswith(':') else ''}"
                for cell in cells
            ]
        elif row_index >= 2 and len(cells) == 9:
            for index in (6, 8):
                try:
                    float(cells[index])
                except ValueError:
                    continue
                cells[index] = "<number>"
        rows.append(cells)
    return rows


def _assert_regression_snapshot(actual, expected, path=()):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), "/".join(path)
        assert set(actual) == set(expected), "/".join(path)
        for key, value in expected.items():
            _assert_regression_snapshot(actual[key], value, (*path, str(key)))
    elif isinstance(expected, list):
        assert isinstance(actual, list), "/".join(path)
        assert len(actual) == len(expected), "/".join(path)
        for index, value in enumerate(expected):
            _assert_regression_snapshot(actual[index], value, (*path, str(index)))
    elif len(path) == 2 and path[0] == "tables":
        assert _table_structure(actual) == _table_structure(expected), "/".join(path)
    elif isinstance(expected, float) and path[0] in {"results", "groups"}:
        assert actual == pytest.approx(expected, abs=1e-4, rel=1e-4), "/".join(path)
    else:
        assert actual == expected, "/".join(path)


def test_cpu_regression_results_and_table(monkeypatch):
    observed = _pin_dataset_revisions(monkeypatch)
    payload = _evaluate_cpu(["arc_easy", "lambada_openai", "wikitext"])
    actual = {
        section: payload.get(section, {})
        for section in (
            "results",
            "groups",
            "versions",
            "n-shot",
            "n-samples",
            "higher_is_better",
        )
    }
    actual["tables"] = {"results": make_table(payload).strip()}

    _assert_regression_snapshot(
        actual, _load_regression_snapshot("cpu_regression_expected.json")
    )
    assert observed == {
        ("allenai/ai2_arc", "ARC-Easy"): DATASET_REVISIONS["allenai/ai2_arc"],
        ("EleutherAI/lambada_openai", "default"): DATASET_REVISIONS[
            "EleutherAI/lambada_openai"
        ],
        ("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1"): (
            DATASET_REVISIONS["EleutherAI/wikitext_document_level"]
        ),
    }


def test_cpu_group_results_and_tables(monkeypatch, tmp_path):
    observed = _pin_dataset_revisions(monkeypatch)
    group_path = tmp_path / "regression_arc.yaml"
    group_path.write_text(
        """\
group: regression_arc
task:
  - arc_challenge
  - arc_easy
aggregate_metric_list:
  - metric: acc
    aggregation: mean
    weight_by_size: true
  - metric: acc_norm
    aggregation: mean
    weight_by_size: true
metadata:
  version: 1.0
""",
        encoding="utf-8",
    )
    task_manager = tasks.TaskManager(include_path=tmp_path)
    payload = _evaluate_cpu(
        ["regression_arc"],
        task_manager=task_manager,
    )

    assert observed == {
        ("allenai/ai2_arc", "ARC-Challenge"): DATASET_REVISIONS["allenai/ai2_arc"],
        ("allenai/ai2_arc", "ARC-Easy"): DATASET_REVISIONS["allenai/ai2_arc"],
    }
    actual = {
        section: payload.get(section, {})
        for section in (
            "results",
            "groups",
            "versions",
            "n-shot",
            "n-samples",
            "higher_is_better",
            "group_subtasks",
        )
    }
    actual["tables"] = {
        "results": make_table(payload).strip(),
        "groups": make_table(payload, "groups").strip(),
    }
    _assert_regression_snapshot(
        actual, _load_regression_snapshot("cpu_group_regression_expected.json")
    )
