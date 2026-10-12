from pathlib import Path

import pytest

from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.utils import apply_template


TASKS_DIR = Path(__file__).parents[1] / "lm_eval" / "tasks"


@pytest.mark.parametrize(
    "config_path, doc",
    [
        (
            "arc/arc_challenge_chat.yaml",
            {"question": "Q?", "choices": {"text": ["a", "b", "c", "d"]}},
        ),
        ("jsonschema_bench/jsonschema_bench_easy.yaml", {"json_schema": "{}"}),
        ("jsonschema_bench/jsonschema_bench_medium.yaml", {"json_schema": "{}"}),
        ("jsonschema_bench/jsonschema_bench_hard.yaml", {"json_schema": "{}"}),
    ],
)
def test_doc_to_text_renders_real_newlines(config_path, doc):
    r"""A single-quoted YAML '\n' reaches the model as a backslash and an 'n'."""
    config = load_yaml(TASKS_DIR / config_path, resolve_func=False)
    prompt = apply_template(config["doc_to_text"], doc)

    assert "\\n" not in prompt
    assert "\n" in prompt
