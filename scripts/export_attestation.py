import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "lm-eval-attestation-v1"


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj)).hexdigest()


def _sha256_jsonl(paths: list[Path]) -> str | None:
    if not paths:
        return None

    digest = hashlib.sha256()
    for path in sorted(paths):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                digest.update(_canonical_json(json.loads(line)))
                digest.update(b"\n")
    return digest.hexdigest()


def _latest_result_file(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = sorted(path.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.json files found under {path}")
    return candidates[-1]


def _result_date_id(result_path: Path) -> str | None:
    stem = result_path.stem
    if not stem.startswith("results_"):
        return None
    return stem.removeprefix("results_")


def _default_sample_paths(result_path: Path) -> list[Path]:
    date_id = _result_date_id(result_path)
    pattern = f"samples_*_{date_id}.jsonl" if date_id else "samples_*.jsonl"
    return sorted(result_path.parent.glob(pattern))


def _metric_stderr_key(metric: str) -> str:
    if "," in metric:
        name, filter_name = metric.split(",", 1)
        return f"{name}_stderr,{filter_name}"
    return f"{metric}_stderr"


def _is_stderr_metric(metric: str) -> bool:
    name = metric.split(",", 1)[0]
    return name.endswith("_stderr") or name == "stderr"


def _canonical_results(results_json: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_id, metrics in sorted(results_json.get("results", {}).items()):
        if not isinstance(metrics, dict):
            continue
        for metric, value in sorted(metrics.items()):
            if metric == "alias" or _is_stderr_metric(metric):
                continue
            row = {
                "task_id": task_id,
                "metric": metric,
                "value": value,
            }
            stderr_key = _metric_stderr_key(metric)
            if stderr_key in metrics:
                row["stderr"] = metrics[stderr_key]
            rows.append(row)
    return rows


def _model_id(results_json: dict[str, Any]) -> str | None:
    if results_json.get("model_name"):
        return results_json["model_name"]

    config = results_json.get("config")
    if isinstance(config, dict):
        for key in ("model", "model_args", "pretrained", "model_name"):
            if config.get(key):
                return str(config[key])
    return None


def build_attestation(
    result_path: Path,
    sample_paths: list[Path] | None = None,
) -> dict[str, Any]:
    result_path = _latest_result_file(result_path)
    samples = _default_sample_paths(result_path) if sample_paths is None else sample_paths

    with result_path.open(encoding="utf-8") as f:
        results_json = json.load(f)

    results = _canonical_results(results_json)
    tasks = sorted({row["task_id"] for row in results})
    model_id = _model_id(results_json)
    results_payload = {
        "model_id": model_id,
        "tasks": tasks,
        "results": results,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "tasks": tasks,
        "results": results,
        "results_sha256": _sha256_json(results_payload),
        "outputs_sha256": _sha256_jsonl(samples),
        "source": {
            "results_path": str(result_path),
            "sample_paths": [str(path) for path in sorted(samples)],
            "git_hash": results_json.get("git_hash"),
            "task_hashes": results_json.get("task_hashes", {}),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a compact, canonical attestation for lm-eval results. "
            "The results hash covers headline scores; the outputs hash covers "
            "per-sample logs when samples_*.jsonl files are available."
        )
    )
    parser.add_argument(
        "result_path",
        type=Path,
        help="Path to a results_*.json file or an output directory containing results_*.json files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the attestation JSON. Defaults to stdout.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional explicit sample JSONL files. By default, matching samples_*_<date>.jsonl "
            "files next to the selected results file are used when present."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attestation = build_attestation(args.result_path, args.samples)
    dumped = json.dumps(attestation, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dumped + "\n", encoding="utf-8")
    else:
        print(dumped)


if __name__ == "__main__":
    main()
