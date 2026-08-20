"""Distribute pinned lm-eval shards over the booked B200 Ray cluster."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lm_eval.tasks import TaskManager
from lm_eval.tasks._index import Kind


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = Path(__file__).with_name("suite.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("smoke", "non_code", "python_code", "multiple", "all"),
        required=True,
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--workers-per-model", type=int, default=8)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--allow-unsafe", action="store_true")
    return parser.parse_args()


def resolve_leaves(manager: TaskManager, selector: str) -> list[str]:
    leaves: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        entry = manager.task_index[name]
        if entry.kind is Kind.TASK:
            leaves.append(name)
        elif entry.kind is Kind.GROUP:
            for child in entry.cfg["task"]:
                visit(child)
        elif entry.kind is Kind.TAG:
            for child in sorted(entry.tags):
                visit(child)
        else:
            raise ValueError(f"Unsupported task registry entry: {name} ({entry.kind})")

    visit(selector)
    return leaves


def leaf_weight(family: dict[str, Any], task: str, leaf_count: int) -> float:
    base = float(family["weight"]) / max(leaf_count, 1)
    if "supergpqa" in task:
        return max(base, 120)
    if task.startswith("bbeh_"):
        return max(base, 4)
    if task.startswith("bbh_"):
        return max(base, 3)
    if task.startswith("mmlu_pro_"):
        return max(base, 5)
    if task.startswith("gpqa_") and ("cot" in task or "generative" in task):
        return max(base, 8)
    if task.startswith("multiple_"):
        return max(base, 10)
    return max(base, 1)


def build_shards(
    suite: dict[str, Any], phase: str, worker_count: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manager = TaskManager()
    if phase == "smoke":
        families = [
            {
                "name": "arc_easy",
                "selector": "arc_easy",
                "phase": "non_code",
                "compute_bpb": True,
                "weight": 1,
            }
        ]
    else:
        phases = {"non_code", "python_code", "multiple"} if phase == "all" else {phase}
        families = [family for family in suite["families"] if family["phase"] in phases]

    weighted_tasks: list[tuple[float, str, bool]] = []
    task_families: dict[str, str] = {}
    for family in families:
        leaves = resolve_leaves(manager, family["selector"])
        compute_bpb = family.get("compute_bpb", True)
        for leaf in leaves:
            previous = task_families.setdefault(leaf, family["name"])
            if previous != family["name"]:
                raise ValueError(
                    f"Task {leaf} belongs to both {previous} and {family['name']}"
                )
            weighted_tasks.append(
                (leaf_weight(family, leaf, len(leaves)), leaf, compute_bpb)
            )

    # BPB support changes the request graph, so never mix BPB and non-BPB
    # tasks in a shard. Greedy bin packing keeps the long generative leaves
    # spread across the available model replicas.
    tasks_by_mode = {
        compute_bpb: sorted(
            (item for item in weighted_tasks if item[2] is compute_bpb), reverse=True
        )
        for compute_bpb in (True, False)
    }
    active_modes = [mode for mode, tasks in tasks_by_mode.items() if tasks]
    allocations = {mode: 1 for mode in active_modes}
    target_workers = min(worker_count, len(weighted_tasks))
    while sum(allocations.values()) < target_workers:
        candidates = [
            mode
            for mode in active_modes
            if allocations[mode] < len(tasks_by_mode[mode])
        ]
        if not candidates:
            break
        mode = max(
            candidates,
            key=lambda candidate: (
                sum(item[0] for item in tasks_by_mode[candidate])
                / allocations[candidate]
            ),
        )
        allocations[mode] += 1

    shards: list[dict[str, Any]] = []
    for compute_bpb in active_modes:
        tasks = tasks_by_mode[compute_bpb]
        bins = [
            {"weight": 0.0, "tasks": [], "compute_bpb": compute_bpb}
            for _ in range(allocations[compute_bpb])
        ]
        for weight, task, _ in tasks:
            target = min(bins, key=lambda item: item["weight"])
            target["tasks"].append(task)
            target["weight"] += weight
        shards.extend(bins)
    return shards, task_families


def run_shard(
    *,
    model: dict[str, str],
    shard_id: str,
    tasks: list[str],
    compute_bpb: bool,
    results_dir: str,
    limit: float | None,
    allow_unsafe: bool,
    python_executable: str,
    data_parallel_size: int = 1,
) -> dict[str, Any]:
    output = Path(results_dir) / model["name"].replace("/", "__") / f"{shard_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(ROOT / "scripts/olmes_bpb_eval/run_eval.py"),
        "--model",
        model["name"],
        "--revision",
        model["revision"],
        "--tasks",
        ",".join(tasks),
        "--output",
        str(output),
    ]
    if compute_bpb:
        command.append("--compute-bpb")
    if allow_unsafe:
        command.append("--confirm-run-unsafe-code")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if data_parallel_size > 1:
        command.extend(["--data-parallel-size", str(data_parallel_size)])

    log_path = output.with_suffix(".log")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
            command,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = {
        "model": model["name"],
        "shard_id": shard_id,
        "tasks": tasks,
        "compute_bpb": compute_bpb,
        "data_parallel_size": data_parallel_size,
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
        "output": str(output),
        "log": str(log_path),
    }
    if completed.returncode:
        raise RuntimeError(json.dumps(status))
    return status


def main() -> None:
    args = parse_args()
    if args.workers_per_model < 1:
        raise ValueError("--workers-per-model must be positive")
    if args.phase in {"python_code", "multiple", "all"} and not args.allow_unsafe:
        raise ValueError(
            f"Phase {args.phase!r} contains executable-code metrics; pass "
            "--allow-unsafe only inside the documented external sandbox setup"
        )
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    shards, task_families = build_shards(suite, args.phase, args.workers_per_model)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    launch_manifest = {
        "suite": suite,
        "phase": args.phase,
        "limit": args.limit,
        "workers_per_model": args.workers_per_model,
        "task_families": task_families,
        "shards": shards,
    }
    (args.results_dir / f"launch-{args.phase}.json").write_text(
        json.dumps(launch_manifest, indent=2) + "\n", encoding="utf-8"
    )

    import ray

    propagated_names = (
        "HF_HOME",
        "HF_TOKEN_PATH",
        "LM_EVAL_PYTHON_EXECUTOR",
        "LM_EVAL_MULTIPLE_EXECUTOR",
    )
    propagated_env = {
        name: os.environ[name] for name in propagated_names if os.environ.get(name)
    }
    ray.init(
        address="auto",
        log_to_driver=True,
        runtime_env={"env_vars": propagated_env} if propagated_env else None,
    )
    remote_run_shard = ray.remote(num_gpus=1, num_cpus=8, max_calls=1)(run_shard)
    refs = []
    for model in suite["models"]:
        for index, shard in enumerate(shards):
            refs.append(
                remote_run_shard.remote(
                    model=model,
                    shard_id=f"{args.phase}-{index:02d}",
                    tasks=shard["tasks"],
                    compute_bpb=shard["compute_bpb"],
                    results_dir=str(args.results_dir),
                    limit=args.limit,
                    allow_unsafe=args.allow_unsafe,
                    python_executable=args.worker_python,
                )
            )

    completed_statuses = []
    while refs:
        done, refs = ray.wait(refs, num_returns=1, timeout=30)
        if not done:
            print(f"Waiting for {len(refs)} shard(s)...", flush=True)
            continue
        status = ray.get(done[0])
        completed_statuses.append(status)
        print(json.dumps(status, sort_keys=True), flush=True)

    (args.results_dir / f"completed-{args.phase}.json").write_text(
        json.dumps(completed_statuses, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
