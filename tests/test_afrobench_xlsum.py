from pathlib import Path

import pytest
import yaml

from lm_eval.utils import apply_template


XLSUM_DIR = Path(__file__).parent.parent / "lm_eval" / "tasks" / "afrobench" / "xlsum"
TASK_YAMLS = sorted(XLSUM_DIR.glob("prompt_*/xlsum_*.yaml"))


def test_task_yamls_found():
    assert len(TASK_YAMLS) == 36


@pytest.mark.parametrize("path", TASK_YAMLS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_prompt_includes_the_document(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompt = apply_template(
        config["doc_to_text"], {"text": "Body of the article.", "summary": "x"}
    )

    assert "Body of the article." in prompt
    # single-quoted YAML keeps "\n" as two characters instead of a newline
    assert "\\n" not in prompt
