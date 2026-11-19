from pathlib import Path

import pytest

from lm_eval.models.utils import postprocess_generated_text
from lm_eval.tasks._yaml_loader import load_yaml


@pytest.mark.parametrize(
    "answer",
    [
        "The paper uses two datasets.",
        "First, the source corpus.\nSecond, the annotated corpus.",
        "First, the source corpus.\n\nSecond, the annotated corpus.",
    ],
    ids=["single-line", "multiline", "paragraph-break"],
)
def test_qasper_freeform_stops_at_next_paper_title(answer: str) -> None:
    task_path = (
        Path(__file__).resolve().parents[1]
        / "lm_eval"
        / "tasks"
        / "qasper"
        / "freeform.yaml"
    )
    config = load_yaml(task_path)
    generated = answer + "\n\nTITLE: The next paper"
    prediction = postprocess_generated_text(
        generated, config["generation_kwargs"]["until"], think_end_token=None
    )

    assert prediction == answer
    assert config["metric_list"][0]["metric"]([prediction], [answer]) == 1.0
