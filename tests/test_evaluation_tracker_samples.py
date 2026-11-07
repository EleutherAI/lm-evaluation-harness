"""Sample serialization must not change records retained by the caller."""

import json

import pytest

from lm_eval.loggers.evaluation_tracker import EvaluationTracker


def _make_sample(kind):
    sample = {"doc_id": 0, "doc": {"question": "Prompt"}, "target": 7}
    if kind == "generation":
        sample.update(
            arguments=[("Prompt", {"until": ["stop"], "max_gen_toks": 4})],
            resps=[["Answer"]],
            filtered_resps=["Answer"],
        )
    else:
        sample.update(
            arguments=[("Prompt", "Choice")],
            resps=[[(-0.25, True)]],
            filtered_resps=[(-0.25, True)],
        )
    return sample


def _make_tracker(output_path):
    tracker = EvaluationTracker(output_path=str(output_path))
    tracker.general_config_tracker.log_experiment_args(
        model_source="dummy",
        model_args={"model": "sample-model"},
        system_instruction=None,
        chat_template=None,
        fewshot_as_multiturn=False,
    )
    tracker.save_results_aggregated({})
    return tracker


def _read_samples(output_path):
    directory = output_path.parent if output_path.suffix == ".json" else output_path
    (path,) = directory.rglob("samples_*.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("kind", ["generation", "multiple_choice"])
@pytest.mark.parametrize("file_output", [False, True])
def test_saving_samples_preserves_input_and_serialized_schema(
    tmp_path, kind, file_output
):
    output_path = tmp_path / "results.json" if file_output else tmp_path
    tracker = _make_tracker(output_path)
    sample = _make_sample(kind)
    original_values = dict(sample)

    tracker.save_results_samples("task", [sample])

    (saved,) = _read_samples(output_path)
    assert saved["target"] == "7"
    assert saved["arguments"]["gen_args_0"]["arg_0"] == "Prompt"
    if kind == "generation":
        assert saved["arguments"]["gen_args_0"]["arg_1"] == {
            "until": ["stop"],
            "max_gen_toks": 4,
        }
        assert saved["resps"] == [["Answer"]]
        assert saved["filtered_resps"] == ["Answer"]
    else:
        assert saved["arguments"]["gen_args_0"]["arg_1"] == "Choice"
        assert saved["resps"] == [[["-0.25", "True"]]]
        assert saved["filtered_resps"] == [["-0.25", "True"]]

    assert sample == _make_sample(kind)
    for key, value in original_values.items():
        assert sample[key] is value


@pytest.mark.parametrize("kind", ["generation", "multiple_choice"])
def test_reusing_samples_with_another_tracker_writes_equal_records(tmp_path, kind):
    sample = _make_sample(kind)
    paths = [tmp_path / "first", tmp_path / "second"]

    for path in paths:
        _make_tracker(path).save_results_samples("task", [sample])

    assert _read_samples(paths[0]) == _read_samples(paths[1])


@pytest.mark.parametrize("kind", ["generation", "multiple_choice"])
def test_repeated_sample_reference_writes_equal_rows(tmp_path, kind):
    sample = _make_sample(kind)
    tracker = _make_tracker(tmp_path)

    tracker.save_results_samples("task", [sample, sample])

    first, second = _read_samples(tmp_path)
    assert first == second


def test_noncopyable_document_values_remain_serializable(tmp_path):
    class OpaqueValue:
        def __str__(self):
            return "opaque-value"

        def __deepcopy__(self, memo):
            raise TypeError("This value cannot be deep-copied")

    opaque = OpaqueValue()
    sample = _make_sample("generation")
    sample["doc"]["opaque"] = opaque

    _make_tracker(tmp_path).save_results_samples("task", [sample])

    (saved,) = _read_samples(tmp_path)
    assert saved["doc"]["opaque"] == "opaque-value"
    assert sample["doc"]["opaque"] is opaque
