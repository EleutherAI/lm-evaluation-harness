"""Offline joins for a complete, previously judged MultiNRC run."""
import json
import os
from pathlib import Path

from lm_eval.judging.core import collect_verdicts, indexed, read_jsonl


def load_dataset(run=None, judge_id=None, **kwargs):
    from datasets import Dataset, DatasetDict

    run = Path(run or os.environ["MULTINRC_RUN"])
    judge_id = judge_id or os.environ["MULTINRC_JUDGE"]
    config = json.loads((run / "config.json").read_text())
    spec = next(stage for stage in config["stages"][1:] if stage["id"] == judge_id)
    verdicts = collect_verdicts(run, spec, read_jsonl(run / judge_id / "verdicts.jsonl"))
    if any(row["verdict"] is None for row in verdicts):
        raise ValueError("Harness accuracy requires complete valid judge coverage")
    lookup = indexed(verdicts)
    rows = [{**row, "judge_verdict": lookup[row["id"]]["verdict"]}
            for row in read_jsonl(run / "rollouts.jsonl")]
    return DatasetDict(test=Dataset.from_list(rows))


def process_results(doc, results):
    if results != [doc["answer"]]:
        raise ValueError("Harness result differs from the frozen candidate answer")
    return {"acc": float(doc["judge_verdict"])}
