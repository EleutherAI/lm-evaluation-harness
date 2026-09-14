"""Text-only LTBv1-eval, with the upstream prompts and all-rules pass rate.

Prompts and verdict parsing adapted from zouharvi/last-translation-benchmark
(MIT): scripts/20b-translate_by_extra_models.py, server/utils.py, and
scripts/41-score_leaderboard.py. Dataset: CC BY 4.0.
"""

import json
import logging
import os
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path


logger = logging.getLogger(__name__)
DEFAULT_JUDGE = "google/gemini-3.1-pro-preview"


def process_docs(dataset):
    """Select the official evaluation tag from the Hugging Face release."""
    dataset = dataset.filter(lambda record: "LTBv1-eval" in (record.get("tags") or []))
    if not len(dataset):
        raise ValueError("The dataset contains no LTBv1-eval examples")
    ids = set()
    for record in dataset:
        if record.get("source_media") or record.get("source_instructions"):
            raise ValueError(
                f"LTBv1-eval example {record['id']} has media or instructions"
            )
        rules = record["verification_rules"]
        if not rules or any(
            not isinstance(rule, str) or not rule.strip() for rule in rules
        ):
            raise ValueError(
                f"Invalid verification rules for LTB example {record['id']}"
            )
        if record["id"] in ids:
            raise ValueError(f"Duplicate LTB example ID: {record['id']}")
        ids.add(record["id"])
    return dataset.select_columns(
        ["id", "source_text", "source_lang", "target_lang", "verification_rules"]
    )


def doc_to_text(doc):
    return (
        f"Translate the following text from {doc['source_lang']} to {doc['target_lang']}. "
        f"Output only the translation and nothing else:\n{doc['source_text']}"
    )


def verification_prompt(source_text, translation, rule):
    return (
        "Your goal is to verify whether a translation fulfills a criterion.\n\n"
        f"Criterion: {rule}\n\nInput: {source_text}\n\n"
        f"Translation to verify: {translation}\n\n"
        "Output only pass or fail and nothing else."
    )


def parse_verdict(response):
    """Match the official leaderboard's permissive pass/fail parser."""
    if response is None:
        return False
    tokens = response.strip().lower().strip(" \t\n\r.,!?\"'*").split()
    last = tokens[-1] if tokens else ""
    if "pass" in last:
        return True
    if "fail" in last:
        return False
    return "pass" in response.lower()


def process_results(doc, results):
    """Defer judging until aggregation so translations can be scored in parallel."""
    translation = results[0]
    response_format = os.environ.get("LTB_RESPONSE_FORMAT", "text")
    if response_format not in ("text", "harmony"):
        raise ValueError(f"Unsupported LTB_RESPONSE_FORMAT: {response_format}")
    if not isinstance(translation, str):
        translation = None
    elif response_format == "harmony":
        # vLLM preserves special tokens. Keep the raw response in harness samples,
        # but score/export only the final channel, never unfinished reasoning.
        for boundary in ("<|channel|>final<|message|>", "<|meta_sep|>final<|im_sep|>"):
            if boundary in translation:
                translation = translation.split(boundary, 1)[1]
                for stop in (
                    "<|return|>",
                    "<|end|>",
                    "<|start|>",
                    "<|fim_suffix|>",
                    "<|im_end|>",
                    "<|im_start|>",
                ):
                    translation = translation.split(stop, 1)[0]
                translation = translation.strip()
                break
        else:
            translation = None
    return {
        "ltb_pass_rate": {
            "id": doc["id"],
            "source_text": doc["source_text"],
            "translation": translation,
            "verification_rules": doc["verification_rules"],
        }
    }


class JudgeLog:
    """Persist predictions before scoring and flush every completed judge call."""

    def __init__(self, items):
        directory = os.environ.get("LTB_OUTPUT_DIR")
        self.directory = Path(
            directory or tempfile.mkdtemp(prefix="ltb-results-", dir=".")
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        # Refuse to mix responses from separate evaluations in the same directory.
        self.handle = (self.directory / "judge_responses.jsonl").open(
            "x", encoding="utf-8"
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    [
                        {"id": item["id"], "translation": item["translation"]}
                        for item in items
                    ],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.directory / "ltb_submission.json")
        except BaseException:
            self.handle.close()
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        logger.info(
            "Saved LTB predictions; judge responses will be logged in %s",
            self.directory.resolve(),
        )

    def write(self, record):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.lock:
            self.handle.write(line)
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.handle.close()


def _score_example(client, model, item, audit=None, stop=None):
    rules = item["verification_rules"]
    if not rules:
        raise ValueError(f"No verification rules for LTB example {item['id']}")
    translation = item["translation"]
    # Empty outputs count as failed attempts, as in the official scorer.
    if not translation:
        if audit is not None:
            audit.write(
                {
                    "event": "example",
                    "id": item["id"],
                    "score": 0,
                    "reason": "empty_translation",
                }
            )
        return 0
    verdicts = []
    for rule_index, rule in enumerate(rules):
        if stop is not None and stop.is_set():
            return None
        request = dict(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": verification_prompt(
                        item["source_text"], translation, rule
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        record = {
            "event": "rule",
            "id": item["id"],
            "rule_index": rule_index,
            "rule": rule,
            "request": request,
        }
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            if audit is not None:
                # Exception messages can contain request credentials; record only
                # the type and HTTP status, never headers or the client object.
                audit.write(
                    {
                        **record,
                        "error_type": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                    }
                )
            raise
        truncated = response.choices[0].finish_reason == "length"
        passed = parse_verdict(response.choices[0].message.content)
        if audit is not None:
            audit.write(
                {
                    **record,
                    "response": response.model_dump(mode="json"),
                    "passed": None if truncated else passed,
                    "truncated": truncated,
                }
            )
        if truncated:
            raise RuntimeError(f"Judge response truncated for LTB example {item['id']}")
        verdicts.append(passed)
    score = int(all(verdicts))
    if audit is not None:
        audit.write({"event": "example", "id": item["id"], "score": score})
    return score


def aggregate_pass_rate(items):
    """Return the micro-average of examples passing every verification rule.

    API failures propagate after SDK retries; they must not silently change the
    denominator or get reported as translation failures.
    """
    items = list(items)
    if not items:
        raise ValueError("Cannot score an empty LTB evaluation")
    api_key = os.environ.get("LTB_JUDGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set LTB_JUDGE_API_KEY to score LTB, or use --predict_only to save translations."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            'Install the LTB judge dependency with pip install -e ".[ltb]"'
        ) from exc

    model = os.environ.get("LTB_JUDGE_MODEL", DEFAULT_JUDGE)
    base_url = os.environ.get("LTB_JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
    workers = int(os.environ.get("LTB_JUDGE_WORKERS", "8"))
    if workers < 1:
        raise ValueError("LTB_JUDGE_WORKERS must be positive")
    logger.info(
        "Scoring %d LTB examples with judge %s at %s", len(items), model, base_url
    )
    with (
        JudgeLog(items) as audit,
        OpenAI(
            api_key=api_key, base_url=base_url, timeout=120.0, max_retries=3
        ) as client,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        stop = threading.Event()

        def score(item):
            try:
                return _score_example(client, model, item, audit, stop=stop)
            except BaseException:
                # Signal other workers immediately, even if an earlier example
                # is still waiting for its judge response.
                stop.set()
                raise

        remaining = iter(items)
        pending = set()
        scores = []
        try:
            for item in islice(remaining, workers):
                pending.add(executor.submit(score, item))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                scores.extend(future.result() for future in done)
                if not stop.is_set():
                    for item in islice(remaining, len(done)):
                        pending.add(executor.submit(score, item))
        finally:
            stop.set()
            for future in pending:
                future.cancel()
            # The executor joins running requests before the client and audit
            # log close. Workers skip any remaining rules after a failure.
    return sum(scores) / len(scores)
