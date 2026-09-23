from pathlib import Path

import pytest

import lm_eval
from lm_eval.models.utils import postprocess_generated_text
from lm_eval.tasks._yaml_loader import load_yaml


BBH_DIR = Path(lm_eval.__file__).parent / "tasks" / "bbh"
TEMPLATES = [
    pytest.param("fewshot/_fewshot_template_yaml", id="fewshot"),
    pytest.param("cot_fewshot/_cot_fewshot_template_yaml", id="cot_fewshot"),
    pytest.param("zeroshot/_zeroshot_template_yaml", id="zeroshot"),
    pytest.param("cot_zeroshot/_cot_zeroshot_template_yaml", id="cot_zeroshot"),
]


def _until(template: str) -> list[str]:
    config = load_yaml(BBH_DIR / template, resolve_func=False)
    return config["generation_kwargs"]["until"]


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize(
    "generation",
    [
        "(Q)",
        "The color of the mug is purple. So the answer is (Q).",
    ],
)
def test_stop_sequences_keep_option_q(template, generation):
    """`reasoning_about_colored_objects` has options (A) to (R); 7 test answers are `(Q)`.

    A bare "Q" stop sequence truncated these answers to "(" before any filter ran.
    """
    assert postprocess_generated_text(generation, _until(template), None) == generation


@pytest.mark.parametrize("template", TEMPLATES)
def test_stop_sequences_still_stop_at_next_question(template):
    # A single newline, so it is the "Q:" stop (not "\n\n") that has to fire.
    generation = "So the answer is (B).\nQ: Today is Christmas Eve of 1937."
    stopped = postprocess_generated_text(generation, _until(template), None)
    assert stopped.startswith("So the answer is (B).")
    assert "Christmas" not in stopped
