"""python -m lm_eval.judging {prepare,judge,report} --config CONFIG."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from .core import atomic_json, prepare, read_jsonl, summarize


def resource_metrics(run, spec):
    directory = run / spec["id"]
    elapsed, job_ids = 0.0, set()
    for path in directory.glob("part-*/execution.json"):
        metrics = json.loads(path.read_text())
        elapsed += metrics["elapsed_seconds"]
        job_ids.add(metrics["job_id"])
    peak = None
    for job_id in job_ids:
        for path in (run / "logs").glob(f"gpu-usage-{job_id}*.csv"):
            with path.open() as stream:
                for row in csv.DictReader(stream, skipinitialspace=True):
                    try:
                        used = float(row["memory.used [MiB]"].split()[0])
                        peak = max(peak or 0, used)
                    except (KeyError, ValueError):
                        continue
    outputs = read_jsonl(directory / "verdicts.jsonl")
    reported_tokens = [row.get("completion_tokens") for row in outputs]
    tokens = sum(n for n in reported_tokens if isinstance(n, int))
    return {"inference_wall_seconds": elapsed,
            "allocated_gpu_hours_during_inference": elapsed * spec["nodes"] * 4 / 3600,
            "peak_gpu_memory_mib": peak,
            "rows_per_second_including_startup": len(outputs) / elapsed if elapsed else None,
            "reported_completion_tokens": tokens,
            "completion_tokens_per_second_including_startup": tokens / elapsed if elapsed else None,
            "token_usage_coverage": sum(isinstance(n, int) for n in reported_tokens),
            "note": "Includes smoke/server startup; GPU hours exclude replay and scheduler overhead. Token totals use terminal attempts only."}
from .runner import execute


def report(config):
    run = Path(config["output"])
    rollouts = read_jsonl(run / "rollouts.jsonl")
    if not rollouts:
        raise ValueError("No frozen rollouts yet")
    summaries, lookups = {}, {}
    for spec in config["stages"][1:]:
        rows = read_jsonl(run / spec["id"] / "verdicts.jsonl")
        summaries[spec["id"]] = summarize(rollouts, rows)
        summaries[spec["id"]]["resources"] = resource_metrics(run, spec)
        summaries[spec["id"]]["configuration"] = spec
        lookups[spec["id"]] = {r["id"]: r["verdict"] for r in rows if r["verdict"] is not None}
    comparisons = {}
    for left, right in combinations(lookups, 2):
        shared = sorted(lookups[left].keys() & lookups[right].keys())
        disagreements = [key for key in shared if lookups[left][key] != lookups[right][key]]
        comparisons[left + "/" + right] = {"shared": len(shared),
            "disagreement_rate": len(disagreements) / len(shared) if shared else None,
            "disagreement_ids": disagreements}
    mode_deltas = {}
    for prefix, low in (("gpt-oss-20b", "low"), ("qwen38-27b", "off"),
                        ("gpt-oss-120b", "low"), ("deepseek-v4-flash", "off")):
        a, b = summaries.get(prefix + "-" + low), summaries.get(prefix + "-high")
        if a and b and a["accuracy"] is not None and b["accuracy"] is not None:
            mode_deltas[prefix] = {"comparison": f"high minus {low}", "accuracy_delta": b["accuracy"] - a["accuracy"]}
    output = {"judges": summaries, "comparisons": comparisons, "mode_deltas": mode_deltas,
              "complete": all(s["failures"] == 0 for s in summaries.values()),
              "interpretation": "Accuracy scores the candidate model; judge reliability requires human labels."}
    atomic_json(run / "report.json", output)
    lines = ["# MultiNRC judge comparison", "", "| Judge | Scored | Accuracy | Failures |",
             "|---|---:|---:|---:|"]
    for name, summary in summaries.items():
        score = f"{summary['accuracy']:.4f}" if summary["accuracy"] is not None else "pending"
        lines.append(f"| {name} | {summary['scored']} | {score} | {summary['failures']} |")
    lines += ["", output["interpretation"]]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "judge", "report"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", type=int, default=0,
                        help="0 generates rollouts; subsequent indices select judge configurations")
    parser.add_argument("--seconds", type=int, default=6300)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "prepare":
        result = prepare(config["dataset"], config["output"], config["stages"][0])
    elif args.command == "judge":
        result = execute(config, args.stage, args.seconds)
    else:
        result = report(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
