"""Helpers for the `collie` task."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any

    import datasets


_COLLIE_URL = "https://collie-benchmark.github.io/data/all_data.dill"


def _download_dill() -> str:
    """Return a local path to the official COLLIE dill, downloading if absent."""
    cache = os.path.join(
        os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/lm_eval")),
        "collie",
        "all_data.dill",
    )
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        urllib.request.urlretrieve(_COLLIE_URL, cache)  # fails explicitly if absent
    return cache


def load_dill(path: str) -> dict[str, list[dict[str, Any]]]:
    """Deserialize the COLLIE dill, returning ``dict[str, list[record]]``.

    Each record is a dict with keys: ``example``, ``metadata``, ``targets``,
    ``constraint`` (a live constraint object), and ``prompt``.

    Note:
    Constraint classes are pickled under the path ``src.constraints``;
    defined in the upstream repo ``github.com/stanford-nlp/collie``.
    We redirect to the our version of `constraints.py` by subclassing
    ``dill.Unpickler``.
    """
    import importlib

    import dill

    vendored = importlib.import_module("lm_eval.tasks.collie.constraints")

    # Subclassing dill.Unpickler to redirect constraint class lookups to the vendored module.
    class _Unpickler(dill.Unpickler):  # noqa: S301
        def find_class(self, module: str, name: str) -> Any:
            if module == "src.constraints":
                return getattr(vendored, name)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return _Unpickler(f).load()


def load_dataset(**kwargs: Any) -> dict[str, datasets.Dataset]:
    """Build the COLLIE table for use as an lm-eval ``custom_dataset`` loader.

    Downloads the official benchmark dill, flattens its ``dict[str, list]`` of
    constraint buckets into rows, and returns a single eval split. Heterogeneous
    fields are JSON-encoded so the Arrow schema is homogeneous; the live
    constraint is dill-serialized (by reference, ~340 B) and base64-encoded so
    docs stay JSON-safe for ``--log_samples``.
    """
    import datasets
    import dill

    data = load_dill(_download_dill())  # dict[str, list[record]]
    rows = []
    for bucket, records in data.items():
        for r in records:
            rows.append(
                {
                    "bucket": bucket,
                    "prompt": r["prompt"],
                    "example": r["example"],
                    "targets": json.dumps(r["targets"]),
                    "metadata": json.dumps(r["metadata"]),
                    "constraint": base64.b64encode(
                        dill.dumps(r["constraint"], byref=True)
                    ).decode(),
                }
            )
    return {"test": datasets.Dataset.from_list(rows)}


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, bool]:
    """Score a generation against its COLLIE constraint (pass-rate accuracy)."""
    import dill

    constraint = dill.loads(base64.b64decode(doc["constraint"]))  # noqa: S301
    targets = json.loads(doc["targets"])
    try:
        passed = bool(constraint.check(results[0], targets))
    except Exception:  # noqa: BLE001
        passed = False
    return {"acc": passed}
