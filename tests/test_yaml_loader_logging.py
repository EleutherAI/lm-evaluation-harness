import logging
from pathlib import Path

from lm_eval.tasks._yaml_loader import load_yaml


def test_external_function_load_logs_warning(tmp_path: Path, caplog) -> None:
    """Resolving a !function from outside the package logs the executed path."""
    (tmp_path / "utils.py").write_text(
        "def make_doc(doc):\n    return doc\n",
        encoding="utf-8",
    )
    (tmp_path / "task.yaml").write_text(
        "task: logging_probe\n"
        "dataset_path: json\n"
        "process_docs: !function utils.make_doc\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="lm_eval.tasks._yaml_loader"):
        cfg = load_yaml(tmp_path / "task.yaml", resolve_func=True)
    assert callable(cfg["process_docs"])
    external = [r for r in caplog.records if "Executing external task code" in r.message]
    assert len(external) == 1
    assert str(tmp_path / "utils.py") in external[0].getMessage()


def test_builtin_function_load_stays_quiet(caplog) -> None:
    """A module imported under the lm_eval.tasks.* namespace logs at debug only."""
    builtin_yaml = Path(__file__).resolve().parents[1] / "lm_eval/tasks/hellaswag/hellaswag.yaml"
    with caplog.at_level(logging.DEBUG, logger="lm_eval.tasks._yaml_loader"):
        cfg = load_yaml(builtin_yaml, resolve_func=True)
    assert "doc_to_text" in cfg
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []
