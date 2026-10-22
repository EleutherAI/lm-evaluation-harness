"""Execute one resumable inference stage using posttraining-inference-engine."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .core import (RUBRIC, atomic_json, collect_verdicts, digest, freeze_rollouts,
                   indexed, judge_inputs, native_verdict, read_jsonl, summarize, write_jsonl)


def read_partial(path):
    """Recover complete lines after a process was stopped during its final write."""
    lines = Path(path).read_bytes().splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise
    return rows


def raw_outputs(directory):
    rows = []
    for path in sorted(Path(directory).glob("part-*/output.jsonl")):
        rows.extend(read_partial(path))
    indexed(rows)
    return rows


def balanced_smoke(rows, docs, per_language=2):
    lookup = indexed(docs)
    counts, chosen = {}, []
    for row in rows:
        language = lookup[row["id"]]["language"]
        if counts.get(language, 0) < per_language:
            chosen.append(row)
            counts[language] = counts.get(language, 0) + 1
    return chosen


def inference_command(config, spec, input_path, output_path, *, judge, dry_run=False):
    engine = Path(config["inference_repo"])
    cmd = ["bash", str(engine / "launch_benchmark.sh"), spec["model"],
           "--profile", spec["profile"], "--profiles-dir", config["profiles_dir"],
           "--instance", f"multinrc-{os.environ.get('SLURM_JOB_ID', 'dryrun')}"]
    if dry_run:
        cmd.append("--dry-run")
    if spec["transport"] == "batch":
        # The current batch worker orchestrator shares one output pathname.
        # Use one TP group to avoid concurrent writers/pruning the resume file.
        cmd += ["--replicas-per-node", "1"]
    cmd += ["--", "--input_file", str(input_path), "--output_file", str(output_path),
            "--thinking-level", spec["level"], "--temperature", str(spec.get("temperature", 0)),
            "--max-new-tokens", str(spec["max_new_tokens"]),
            "--metrics-burn-completions", "0", "--metrics-start-completion", "1"]
    for key in ("top_p", "top_k"):
        if key in spec:
            cmd += ["--" + key.replace("_", "-"), str(spec[key])]
    if spec["transport"] == "http":
        cmd += ["--chat-completions", "--max-inflight", "32", "--request-timeout-s", "1200"]
    else:
        cmd += ["--orchestrate_workers", "1", "--worker_max_restarts", "0"]
    if judge:
        cmd += ["--output-validator", "lm_eval.judging.core:parse_verdict",
                "--max-output-retries", "2"]
    return cmd


def run_process(command, log, timeout, env):
    """Run a process group and clean up on deadline or scheduler termination."""
    started = time.monotonic()
    with Path(log).open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        def stop():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        previous = {}
        def interrupted(signum, frame):
            stop()
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        timed_out = False
        try:
            try:
                code = process.wait(timeout=max(1, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
                stop()
                code = process.returncode
        finally:
            stop()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return {"returncode": code, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started}


def d1_batch(config, spec, rows, output, deadline):
    import torch
    from inference.common.decision_model import DecisionModel

    if not torch.cuda.is_available():
        raise RuntimeError("D1 judging requires an ACC GPU")
    model = DecisionModel(spec["model"], token_budget=32768)
    question = {"type": "choice", "instructions": RUBRIC.replace(
        "Return exactly 1 for correct or 0 for incorrect, with no other text.",
        "Choose correct or incorrect."),
        "criteria": {"correct": "The candidate answer is correct according to the rubric.",
                     "incorrect": "The candidate answer is incorrect according to the rubric."}}
    with Path(output).open("a") as stream, torch.inference_mode():
        for start in range(0, len(rows), 8):
            if time.monotonic() >= deadline:
                break
            batch = rows[start:start + 8]
            answers = model.system_one_batch([(row["state"], {"correctness": question})
                                             for row in batch])
            if len(answers) != len(batch):
                raise ValueError("D1 batch output count mismatch")
            for row, answer in zip(batch, answers):
                if answer.get("usage", {}).get("output_tokens") != 0:
                    raise ValueError("Native D1 generated output tokens")
                decision = answer["answers"]["correctness"]
                verdict = native_verdict(decision)
                stream.write(json.dumps({**row, "output": str(verdict),
                    "decision": decision, "raw_decision": answer,
                    "inference_status": "ok", "inference_attempts": 1,
                    "completion_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0},
                    ensure_ascii=False) + "\n")
                stream.flush()


def execute(config, stage_index, seconds=6300):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inference stages must run inside an ACC Slurm allocation")
    run = Path(config["output"])
    spec = config["stages"][stage_index]
    judge = stage_index != 0
    directory = run / spec["id"]
    directory.mkdir(exist_ok=True)
    config_hash = digest({"spec": spec, "manifest": json.loads((run / "manifest.json").read_text())})
    stamp = directory / "config-hash.json"
    if stamp.exists() and json.loads(stamp.read_text()) != config_hash:
        raise ValueError("Stage configuration changed; use a new output directory")
    atomic_json(stamp, config_hash)
    inputs = judge_inputs(run, spec) if judge else read_jsonl(run / "generation-input.jsonl")
    docs = read_jsonl(run / "dataset.jsonl")
    expected = indexed(inputs)
    before = raw_outputs(directory)
    for row in before:
        source = expected[row["id"]]
        for key in ("input_hash",) if not judge else ("answer_hash", "rubric_hash", "judge_hash"):
            if row.get(key) != source[key]:
                raise ValueError(f"Resumed output has incompatible {key}")
    seen = indexed(before)
    pending = [row for row in inputs if row["id"] not in seen]
    deadline = time.monotonic() + seconds
    env = os.environ.copy()
    env["PTIE_EXTRA_PYTHONPATH"] = os.pathsep.join(
        [config["inference_repo"], config["harness"], config["metadata"]])
    env["SGLANG_STATE_DIR"] = str(directory / "server-state")
    env["VISUAL_SDG_RUNTIME_ROOT"] = str(directory / "runtime")
    for key in ("WANDB_MODE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = "offline" if key == "WANDB_MODE" else "1"
        env[key] = value
        env["SINGULARITYENV_" + key] = value
    while pending and time.monotonic() < deadline - 60:
        smoke = not seen
        batch = balanced_smoke(pending, docs) if smoke else pending
        part = directory / f"part-{len(list(directory.glob('part-*'))):04d}"
        part.mkdir()
        write_jsonl(part / "input.jsonl", batch)
        if spec["transport"] == "d1":
            started = time.monotonic()
            d1_batch(config, spec, batch, part / "output.jsonl", deadline - 30)
            metrics = {"elapsed_seconds": time.monotonic() - started, "returncode": 0,
                       "timed_out": time.monotonic() >= deadline - 30}
        else:
            command = inference_command(config, spec, part / "input.jsonl",
                                        part / "output.jsonl", judge=judge)
            atomic_json(part / "command.json", command)
            metrics = run_process(command, part / "inference.log", deadline - time.monotonic() - 30, env)
        atomic_json(part / "execution.json", {**metrics, "job_id": os.environ["SLURM_JOB_ID"]})
        if metrics["returncode"] and not metrics["timed_out"]:
            raise RuntimeError(f"Inference failed; see {part / 'inference.log'}")
        current = raw_outputs(directory)
        seen = indexed(current)
        failures = [row for row in current if row.get("inference_status") not in (
            {"ok"} if judge else {"ok", "discarded_max_length"})]
        if failures:
            atomic_json(directory / "failures.json", failures)
            raise RuntimeError(f"{len(failures)} failed rows; inspect {directory}")
        if smoke:
            if not metrics["timed_out"] and len(current) < len(batch):
                raise RuntimeError("Smoke inference omitted rows")
            atomic_json(directory / "smoke.json", {"ids": [row["id"] for row in current],
                "complete": len(current) == len(batch),
                "empty_outputs": sum(not row.get("output", "").strip() for row in current),
                "truncated_outputs": sum(row.get("inference_status") == "discarded_max_length" for row in current)})
        pending = [row for row in inputs if row["id"] not in seen]
        if metrics["timed_out"]:
            break
        if pending and not smoke:
            raise RuntimeError("Inference exited successfully but omitted rows")
    current = raw_outputs(directory)
    status = {"stage": spec["id"], "expected": len(inputs), "completed": len(current),
              "new_rows": len(current) - len(before), "complete": len(current) == len(inputs)}
    atomic_json(directory / "status.json", status)
    if status["complete"]:
        if judge:
            verdicts = collect_verdicts(run, spec, current)
            write_jsonl(directory / "verdicts.jsonl", verdicts, immutable=True)
            summary = summarize(read_jsonl(run / "rollouts.jsonl"), verdicts)
            atomic_json(directory / "summary.json", summary)
            if summary["failures"]:
                raise RuntimeError("Judge stage has unscored rows")
        else:
            freeze_rollouts(run, current)
    return status
