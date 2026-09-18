"""The bigbench multiple-choice generator decides from a representative row.

Some subtasks mix multiple-choice and free-form examples in one split, so the
first row does not decide for the rest. `utils.filter_multiple_choice` drops
the free-form rows at run time; before this, the generator skipped the whole
subtask whenever row 0 happened to be one of them.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml


GENERATOR = (
    Path(__file__).parent.parent
    / "lm_eval"
    / "tasks"
    / "bigbench"
    / "generate_tasks.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("bigbench_generate_tasks", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generator(monkeypatch, tmp_path):
    module = _load_generator()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "all_subtasks", ["subtask"])
    return module


def _stub(monkeypatch, module, rows):
    def load_dataset(_path, _name):
        return {"default": rows}

    monkeypatch.setattr(module.datasets, "load_dataset", load_dataset)


FREE_FORM = {"targets": ["a free-form answer"], "multiple_choice_targets": []}
CHOICE_ROW = {"targets": ["b"], "multiple_choice_targets": ["a", "b", "c"]}


def _written(tmp_path):
    path = tmp_path / "multiple_choice" / "subtask.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_a_subtask_whose_first_row_is_free_form_still_gets_a_task(
    generator, monkeypatch, tmp_path
):
    _stub(monkeypatch, generator, [FREE_FORM, CHOICE_ROW])
    generator.main()

    written = _written(tmp_path)
    assert written is not None, "the subtask was skipped on the strength of row 0"
    assert written["task"] == "bigbench_subtask_multiple_choice"
    assert written["include"] == "../multiple_choice_template_a_yaml"


def test_a_subtask_with_no_choices_anywhere_is_still_skipped(
    generator, monkeypatch, tmp_path
):
    _stub(monkeypatch, generator, [FREE_FORM, FREE_FORM])
    generator.main()

    assert _written(tmp_path) is None


def test_the_template_follows_the_row_that_has_choices(
    generator, monkeypatch, tmp_path
):
    """Template b resolves the answer through multiple_choice_scores, and is
    chosen when the target is not among the choices.
    """
    unlisted = {"targets": ["z"], "multiple_choice_targets": ["a", "b", "c"]}
    _stub(monkeypatch, generator, [FREE_FORM, unlisted])
    generator.main()

    assert _written(tmp_path)["include"] == "../multiple_choice_template_b_yaml"
