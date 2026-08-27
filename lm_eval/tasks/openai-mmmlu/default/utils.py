import json
from functools import partial
from pathlib import Path

SUBJECTS_PATH = Path(__file__).resolve().parent.parent / "subjects.json"
with SUBJECTS_PATH.open(encoding="utf-8") as f:
    SUBJECTS = json.load(f)


def _normalize_subject_name(name: str) -> str:
    """
    Some MMMLU configs embed CSV filenames in the Subject column (e.g.
    `college_mathematics_test.csv_sw-KE.csv`). Strip the `_test.csv` suffix and
    anything that follows so we always compare against the canonical subject id.
    """
    if not isinstance(name, str):
        return name
    for marker in ("_test.csv", "_test-"):
        idx = name.find(marker)
        if idx != -1:
            return name[:idx]
    return name


def _filter_subject(dataset, subject):
    normalized = subject

    def _predicate(row, target=normalized):
        row_subject = _normalize_subject_name(row["Subject"])
        return row_subject == target

    return dataset.filter(_predicate)


def _register_subject_filters():
    for subject in SUBJECTS:
        globals()[f"process_{subject}"] = partial(_filter_subject, subject=subject)


_register_subject_filters()
