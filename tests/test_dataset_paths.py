"""Static guard: every task's resolved dataset_path must be a namespaced Hub id or a local builder.

huggingface_hub>=1.16 raises HfUriError for single-segment repo ids (e.g. ``gsm8k``)
when datasets resolves files through HfFileSystem. This test needs no network.
Scans both task YAMLs (``dataset_path:``) and task Python classes
(class-level ``DATASET_PATH`` string constants).
"""

import ast
import os
from pathlib import Path

import pytest
import yaml


TASKS_DIR = Path(
    os.environ.get(
        "LMEVAL_TASKS_DIR", Path(__file__).resolve().parents[1] / "lm_eval" / "tasks"
    )
)

# Packaged builders / placeholders that are not Hub repo ids.
LOCAL_BUILDERS = {
    "json",
    "csv",
    "parquet",
    "arrow",
    "text",
    "null",
    "None",
    "pandas",
    "imagefolder",
    "audiofolder",
    "webdataset",
}


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_multi_constructor(
    "", lambda loader, suffix, node: None
)  # ignore !function etc.


def _iter_dataset_paths():
    """Yield (file, dataset_path) for every dataset_path literal, including list-form task entries.

    A YAML file that fails to parse yields a single sentinel entry so the failure is
    reported by name rather than silently skipped.
    """
    for path in sorted(TASKS_DIR.rglob("*")):
        if not path.is_file() or not (
            path.suffix in {".yaml", ".yml"} or path.name.endswith("yaml")
        ):
            continue
        try:
            data = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)  # noqa: S506 - SafeLoader subclass
        except yaml.YAMLError as exc:
            yield path.relative_to(TASKS_DIR), f"<unparsable YAML: {exc}>"
            continue
        data = data if isinstance(data, dict) else {}
        entries = [data]
        if isinstance(
            data.get("task"), list
        ):  # group files such as benchmarks/t0_eval.yaml
            entries += [t for t in data["task"] if isinstance(t, dict)]
        for entry in entries:
            value = entry.get("dataset_path")
            if isinstance(value, str):
                yield path.relative_to(TASKS_DIR), value


def _iter_py_dataset_paths():
    """Yield (file, dataset_path) for every class-level DATASET_PATH string constant."""
    for path in sorted(TASKS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name) and target.id == "DATASET_PATH"
                    for target in stmt.targets
                ):
                    continue
                if isinstance(stmt.value, ast.Constant) and isinstance(
                    stmt.value.value, str
                ):
                    yield path.relative_to(TASKS_DIR), stmt.value.value


def _is_valid(value: str) -> bool:
    if value.startswith("<unparsable"):  # YAML parse failure sentinel
        return False
    if (
        not value or value in LOCAL_BUILDERS or "{" in value
    ):  # unset, local builder or templated value
        return True
    if value.startswith(("/", "./", "hf://")):  # explicit local path or full URI
        return True
    return value.count("/") == 1 and all(value.split("/"))


def test_dataset_paths_are_namespaced():
    bad = [
        f"{file}: dataset_path: {value}"
        for file, value in list(_iter_dataset_paths()) + list(_iter_py_dataset_paths())
        if not _is_valid(value)
    ]
    assert not bad, (
        "Single-segment Hugging Face dataset ids fail with huggingface_hub>=1.16 (HfUriError). "
        "Use the namespaced id the Hub redirects to:\n" + "\n".join(bad)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
