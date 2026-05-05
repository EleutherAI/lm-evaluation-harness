"""Export an attestation bundle from lm-evaluation-harness output.

Reads --output_path / --log_samples artifacts and emits a JSON document
containing SHA256 digests of the headline scores and per-sample outputs.
See attestation.schema.json for the bundle shape and canonical encoding rules.

Stdlib-only. The bundle provides integrity, not authenticity; sign it
separately if you need authentication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any


ATTESTATION_VERSION = 1


class AttestationError(Exception):
    exit_code = 1


class ResultsError(AttestationError):
    exit_code = 3


class SamplesError(AttestationError):
    exit_code = 4


def _format_float(x: float) -> str:
    if not math.isfinite(x):
        raise ValueError(f"non-finite float not allowed: {x!r}")
    if x == 0.0:
        return "0.0"
    s = repr(float(x))
    if "e" in s:
        mantissa, _, exp = s.partition("e")
        if exp.startswith("+"):
            exp = exp[1:]
        sign = "-" if exp.startswith("-") else ""
        digits = exp.lstrip("+-").lstrip("0") or "0"
        s = f"{mantissa}e{sign}{digits}"
    return s


def _canonicalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return _format_float(obj)
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"dict keys must be strings; got {type(k).__name__}")
            out[unicodedata.normalize("NFC", k)] = _canonicalize(v)
        return out
    raise TypeError(f"unsupported type: {type(obj).__name__}")


def canonical_encode(obj: Any) -> bytes:
    return json.dumps(
        _canonicalize(obj),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def derive_model_id(doc: dict) -> str:
    source = doc.get("model_source")
    name = doc.get("model_name")
    if source and name:
        return f"{source}/{name}"

    config = doc.get("config") or {}
    raw_source = config.get("model")
    raw_args = config.get("model_args")
    if raw_source and raw_args:
        for prefix in ("peft", "delta", "pretrained", "model", "path", "engine"):
            if isinstance(raw_args, dict):
                value = raw_args.get(prefix)
                if isinstance(value, str) and value.strip():
                    return f"{raw_source}/{value.strip()}"
            if isinstance(raw_args, str) and f"{prefix}=" in raw_args:
                tail = raw_args.split(f"{prefix}=", 1)[1]
                value = tail.split(",", 1)[0].strip()
                if value:
                    return f"{raw_source}/{value}"
        return str(raw_source)
    return raw_source or name or "unknown"


def _format_metric_value(value: Any) -> str | None:
    if not _is_number(value):
        return None
    return _format_float(float(value))


def extract_results(doc: dict) -> list[dict]:
    entries: list[dict] = []
    for task_id, metrics in (doc.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            metric, sep, filter_key = key.partition(",")
            if not sep or metric.endswith("_stderr"):
                continue
            value_str = _format_metric_value(value)
            if value_str is None:
                continue
            stderr_str = _format_metric_value(
                metrics.get(f"{metric}_stderr,{filter_key}")
            )
            entries.append({
                "task_id": task_id,
                "metric": metric,
                "filter": filter_key,
                "value": value_str,
                "stderr": stderr_str,
            })
    entries.sort(key=lambda e: (e["task_id"], e["metric"], e["filter"]))
    return entries


def derive_tasks(entries: list[dict]) -> list[str]:
    return sorted({e["task_id"] for e in entries})


def derive_meta(doc: dict) -> dict:
    return {
        "git_hash": doc.get("git_hash"),
        "versions": doc.get("versions") or {},
        "n_shot": doc.get("n-shot") or {},
        "date": doc.get("date"),
    }


def _date_id(name: str) -> str:
    stem = name
    for ext in (".jsonl", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem.rsplit("_", 1)[-1]


def _task_from_samples_name(name: str) -> str:
    stem = name
    for ext in (".jsonl", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    parts = stem.split("_", 1)
    if len(parts) < 2:
        raise SamplesError(f"unrecognised samples filename: {name}")
    return parts[1].rsplit("_", 1)[0]


def iter_sample_records(samples_paths: list[Path]) -> Iterator[tuple[str, dict]]:
    for p in samples_paths:
        task_id = _task_from_samples_name(p.name)
        try:
            f = p.open(encoding="utf-8")
        except FileNotFoundError as e:
            raise SamplesError(f"samples file not found: {p}") from e
        except OSError as e:
            raise SamplesError(f"could not read {p}: {e}") from e
        with f:
            for line_no, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise SamplesError(f"{p}:{line_no}: {e}") from e
                yield task_id, record


def compute_outputs_sha256(samples_paths: list[Path]) -> str:
    by_task: dict[str, list[dict]] = {}
    for task_id, record in iter_sample_records(samples_paths):
        by_task.setdefault(task_id, []).append(record)

    h = hashlib.sha256()
    for task_id in sorted(by_task):
        records = by_task[task_id]
        records.sort(key=lambda r: (r.get("doc_id"), r.get("filter", "")))
        seen: set[tuple] = set()
        for r in records:
            if "doc_id" not in r or "filtered_resps" not in r:
                raise SamplesError(
                    f"sample for task {task_id!r} missing doc_id/filtered_resps"
                )
            key = (r["doc_id"], r.get("filter", ""))
            if key in seen:
                raise SamplesError(
                    f"collision in task {task_id!r}: doc_id={r['doc_id']!r}, "
                    f"filter={r.get('filter', '')!r}"
                )
            seen.add(key)
            h.update(canonical_encode({
                "task_id": task_id,
                "doc_id": r["doc_id"],
                "filter": r.get("filter", ""),
                "filtered_resps": r["filtered_resps"],
            }))
            h.update(b"\n")
    return h.hexdigest()


def discover_pair(input_dir: Path) -> tuple[Path, list[Path]]:
    if not input_dir.is_dir():
        raise ResultsError(f"not a directory: {input_dir}")
    results_files = list(input_dir.glob("results_*.json"))
    if not results_files:
        raise ResultsError(f"no results_*.json in {input_dir}")
    latest = max(results_files, key=lambda p: _date_id(p.name))
    return latest, pair_samples_for_results(latest)


def pair_samples_for_results(results_path: Path) -> list[Path]:
    date_id = _date_id(results_path.name)
    return sorted(
        p for p in results_path.parent.glob("samples_*.jsonl")
        if _date_id(p.name) == date_id
    )


def build_attestation(
    results_path: Path,
    samples_paths: list[Path] | None,
    *,
    include_outputs: bool = True,
) -> dict:
    try:
        doc = json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ResultsError(str(e)) from e
    except json.JSONDecodeError as e:
        raise ResultsError(f"{results_path}: {e}") from e

    entries = extract_results(doc)
    if not entries:
        raise ResultsError(f"no scalar metric entries in {results_path}")

    model_id = derive_model_id(doc)
    tasks = derive_tasks(entries)
    claim = {"model_id": model_id, "tasks": tasks, "results": entries}

    if include_outputs and samples_paths:
        outputs_sha = compute_outputs_sha256(samples_paths)
    else:
        outputs_sha = None

    return {
        "harness_attestation_version": ATTESTATION_VERSION,
        "model_id": model_id,
        "tasks": tasks,
        "results": entries,
        "results_sha256": sha256_hex(canonical_encode(claim)),
        "outputs_sha256": outputs_sha,
        "meta": derive_meta(doc),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export an attestation bundle from lm-evaluation-harness output."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input_dir", type=Path,
        help="Directory containing results_*.json (and samples_*.jsonl).",
    )
    src.add_argument(
        "--results", type=Path, help="Explicit results_*.json file.",
    )
    p.add_argument(
        "--samples", nargs="+", type=Path, default=None,
        help="Explicit sample files; only valid with --results.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the bundle. Default: stdout.",
    )
    p.add_argument(
        "--no_samples", action="store_true",
        help="Skip outputs hashing.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples is not None and args.results is None:
        print("--samples requires --results", file=sys.stderr)
        return 2

    try:
        if args.input_dir is not None:
            results_path, samples_paths = discover_pair(args.input_dir)
        else:
            results_path = args.results
            samples_paths = (
                list(args.samples) if args.samples is not None
                else pair_samples_for_results(results_path)
            )
        bundle = build_attestation(
            results_path, samples_paths, include_outputs=not args.no_samples
        )
    except AttestationError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code

    output = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
