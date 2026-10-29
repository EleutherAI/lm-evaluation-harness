import json

import pytest

from lm_eval import utils
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import sanitize_model_name


# 255 is the usual per-component limit on ext4/APFS/NTFS. Anything a path
# component is built from must stay comfortably under it.
FS_COMPONENT_LIMIT = 255


def test_short_name_is_only_sanitized():
    assert sanitize_model_name("EleutherAI/pythia-14m") == "EleutherAI__pythia-14m"


def test_long_name_is_truncated_and_unique():
    a = "model-" + "a" * 400
    b = "model-" + "a" * 399 + "b"
    sa, sb = sanitize_model_name(a), sanitize_model_name(b)

    assert len(sa) <= utils.MAX_MODEL_NAME_LENGTH
    assert len(sb) <= utils.MAX_MODEL_NAME_LENGTH
    # long shared prefix must not collapse two different models together
    assert sa != sb


def test_truncation_is_stable():
    name = "/" + "verylongdirectoryname/" * 12 + "checkpoint"
    assert sanitize_model_name(name) == sanitize_model_name(name)


@pytest.mark.parametrize("length", [0, 1, 119, 120, 121, 1000])
def test_never_exceeds_limit(length):
    assert len(sanitize_model_name("x" * length)) <= FS_COMPONENT_LIMIT


def test_sample_results_saved_with_long_pretrained_path(tmp_path):
    """Regression test: a long `pretrained=` path must not silently drop samples."""
    long_path = "/" + "/".join(["verylongdirectoryname"] * 12) + "/checkpoint"

    tracker = EvaluationTracker(output_path=str(tmp_path))
    tracker.general_config_tracker.log_experiment_args(
        model_source="hf",
        model_args=f"pretrained={long_path},dtype=float32",
        system_instruction=None,
        chat_template=None,
        fewshot_as_multiturn=False,
    )
    # save_results_samples reuses date_id set by save_results_aggregated, so
    # mirror the real call order from the evaluator.
    tracker.save_results_aggregated(results={"results": {"hellaswag": {}}})
    tracker.save_results_samples(
        "hellaswag",
        [
            {
                "doc_id": 0,
                "arguments": [],
                "resps": [["x"]],
                "filtered_resps": ["x"],
                "target": "y",
            }
        ],
    )

    written = list(tmp_path.rglob("samples_*.jsonl"))
    assert written, "sample results were not written to disk"
    assert len(json.loads(written[0].read_text().strip())) > 0
