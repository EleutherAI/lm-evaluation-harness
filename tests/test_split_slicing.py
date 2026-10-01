"""Offline coverage of Hugging Face split expressions in task configuration."""

import datasets
import pytest

from lm_eval import evaluator
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.dummy import DummyLM


@pytest.fixture
def data():
    return datasets.DatasetDict(
        {
            name: datasets.Dataset.from_dict(
                {"row": list(range(size)), "split": [name] * size}
            )
            for name, size in [("train", 12), ("validation", 8), ("test", 10)]
        }
    )


@pytest.fixture
def make_task(data, monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", lambda **kwargs: data)

    def build(**kwargs):
        return ConfigurableTask(
            config={
                "task": "sliced_test",
                "dataset_path": "offline-fixture",
                "training_split": "train",
                "validation_split": "validation",
                "test_split": "test",
                "output_type": "generate_until",
                "doc_to_text": "{{split}} {{row}}",
                "doc_to_target": "{{row}}",
                "num_fewshot": 0,
                "metric_list": [
                    {
                        "metric": "exact_match",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
                **kwargs,
            }
        )

    return build


@pytest.mark.parametrize(
    "field,method,spec,expected",
    [
        ("training_split", "training_docs", "train[1:4]", [1, 2, 3]),
        ("validation_split", "validation_docs", "validation[-2:]", [6, 7]),
        ("test_split", "test_docs", "test[:50%]", list(range(5))),
        ("test_split", "test_docs", "test[50%:]", list(range(5, 10))),
        ("test_split", "test_docs", "test[-20%:]", [8, 9]),
        ("training_split", "training_docs", "train[:0]", []),
        ("test_split", "test_docs", "test[:]+test[:2]", list(range(10)) + [0, 1]),
        ("fewshot_split", "fewshot_docs", "train[:3]", [0, 1, 2]),
    ],
)
def test_split_expression_selects_expected_rows(
    make_task, field, method, spec, expected
):
    task = make_task(**{field: spec})
    assert getattr(task, method)()["row"] == expected


def test_split_concatenation_preserves_order(make_task):
    selected = make_task(test_split="test[2:4]+validation[-2:]").test_docs()
    assert selected["row"] == [2, 3, 6, 7]
    assert selected["split"] == ["test", "test", "validation", "validation"]


@pytest.mark.parametrize(
    "method,name",
    [
        ("training_docs", "train"),
        ("validation_docs", "validation"),
        ("test_docs", "test"),
    ],
)
def test_named_splits_preserve_dataset_identity(make_task, data, method, name):
    assert getattr(make_task(), method)() is data[name]


def test_processing_runs_after_selection(make_task):
    observed = []

    def process(selected):
        observed.append(selected["row"])
        return selected.map(lambda row: {"row": row["row"] + 100})

    task = make_task(test_split="test[2:4]", process_docs=process)
    assert task.test_docs()["row"] == [102, 103]
    assert observed[-1] == [2, 3]


def test_fewshot_processing_runs_after_selection(make_task):
    task = make_task(
        fewshot_config={
            "split": "train[:2]",
            "process_docs": lambda selected: selected.select([1]),
        }
    )
    assert task.fewshot_docs()["row"] == [1]


@pytest.mark.parametrize("spec", ["test[broken]", "missing[:2]", "test[:200%]"])
def test_invalid_split_expressions_raise(make_task, spec):
    with pytest.raises(ValueError):
        make_task(test_split=spec).test_docs()


def test_missing_named_split_keeps_key_error(make_task):
    with pytest.raises(KeyError, match="missing"):
        make_task(test_split="missing").test_docs()


def test_streaming_slicing_reports_unsupported_mode(make_task, data):
    data["test"] = data["test"].to_iterable_dataset()
    with pytest.raises(ValueError, match="materialized datasets"):
        make_task(test_split="test[:2]").test_docs()


def test_evaluator_counts_only_selected_rows(make_task):
    task = make_task(test_split="test[2:5]")
    model = DummyLM()
    model.tokenizer = None
    result = evaluator.simple_evaluate(model=model, tasks=[task], bootstrap_iters=0)
    assert result["n-samples"]["sliced_test"] == {"original": 3, "effective": 3}
    assert [sample["doc"]["row"] for sample in result["samples"]["sliced_test"]] == [
        2,
        3,
        4,
    ]


def test_same_sliced_fewshot_pool_excludes_evaluation_document(make_task):
    task = make_task(test_split="test[:4]", fewshot_split="test[:4]", num_fewshot=2)
    task.set_fewshot_seed(42)
    for doc in task.test_docs():
        context = task.fewshot_context(doc=doc, num_fewshot=2)
        assert context.count(f"test {doc['row']}") == 1
