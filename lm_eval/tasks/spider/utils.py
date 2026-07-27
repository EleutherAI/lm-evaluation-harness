import functools
import os
import re
import sqlite3

import datasets
from huggingface_hub import hf_hub_download


# The user must set this to the HF dataset repo that hosts the Spider database
# files and precomputed schema.json (see scripts/prepare_spider_hf_dataset.py).
# Set via env var so the task doesn't hardcode a specific username's repo.
SPIDER_HF_REPO = os.environ.get("SPIDER_HF_REPO", "Nick888999/spider-eval")

_SQL_EXEC_TIMEOUT_SECONDS = 10


@functools.lru_cache(maxsize=1)
def _load_schema_map() -> dict:
    """db_id -> CREATE TABLE-style schema text, precomputed offline from tables.json."""
    path = hf_hub_download(
        repo_id=SPIDER_HF_REPO, repo_type="dataset", filename="schema.json"
    )
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


@functools.cache
def _db_path(db_id: str) -> str:
    return hf_hub_download(
        repo_id=SPIDER_HF_REPO,
        repo_type="dataset",
        filename=f"database/{db_id}/{db_id}.sqlite",
    )


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    schema_map = _load_schema_map()

    def _process(doc):
        return {
            "db_id": doc["db_id"],
            "question": doc["question"],
            "gold_sql": doc["query"],
            "schema": schema_map[doc["db_id"]],
        }

    return dataset.map(_process)


def doc_to_text(doc) -> str:
    return (
        f"{doc['schema']}\n"
        f"-- Using valid SQLite, answer the following question for the tables "
        f"provided above.\n"
        f"-- {doc['question']}\n"
        f"SELECT"
    )


_SQL_TAG_RE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_sql(generated: str) -> str:
    """Model output is appended after our `SELECT` prefix; also handle the
    case where a chat model re-emits a full fenced query instead.
    """
    match = _SQL_TAG_RE.search(generated)
    if match:
        return match.group(1).strip()
    generated = generated.strip()
    if generated.lower().startswith("select"):
        return generated
    return "SELECT " + generated


def _run_query(db_path: str, sql: str):
    con = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, timeout=_SQL_EXEC_TIMEOUT_SECONDS
    )
    try:
        con.execute("PRAGMA query_only = ON;")
        cur = con.execute(sql)
        rows = cur.fetchall()
    finally:
        con.close()
    return rows


def _results_match(pred_rows, gold_rows, order_matters: bool) -> bool:
    if order_matters:
        return pred_rows == gold_rows
    return sorted(map(str, pred_rows)) == sorted(map(str, gold_rows))


def process_results(doc, results) -> dict:
    generated = results[0]
    predicted_sql = _extract_sql(generated)
    db_path = _db_path(doc["db_id"])

    try:
        gold_rows = _run_query(db_path, doc["gold_sql"])
    except sqlite3.Error:
        # Gold query itself failed to execute (shouldn't normally happen) --
        # exclude from scoring by treating as a miss rather than crashing the run.
        return {"execution_accuracy": 0.0}

    try:
        pred_rows = _run_query(db_path, predicted_sql)
    except sqlite3.Error:
        return {"execution_accuracy": 0.0}

    order_matters = "order by" in doc["gold_sql"].lower()
    correct = _results_match(pred_rows, gold_rows, order_matters)
    return {"execution_accuracy": 1.0 if correct else 0.0}
