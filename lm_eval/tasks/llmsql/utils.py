"""LLMSQL 2.0: execution-based scoring of generated SQLite queries.

The reference implementation of this protocol is the `llmsql` package
(https://github.com/LLMSQL/llmsql-benchmark): the SQL is taken from the last
```sql block of the completion (falling back to the first WITH/SELECT
statement), executed on the benchmark database, and compared with the
verified reference answer by a lenient execution match.
"""

import json
import math
import re
import sqlite3
import time
from functools import lru_cache


_DB_REPO = "llmsql-bench/llmsql-2.0"
_SQL_TIMEOUT = 10.0
_UNIT = r"(?:million|billion|thousand|m|bn|k|%|km|kg|mph|km/h|cm|mm|metres|meters|ft|lbs?|mhz|kw|s)"


@lru_cache(maxsize=1)
def _db_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_DB_REPO, "sqlite_tables.db", repo_type="dataset")


def extract_sql(text):
    """SQL from the last ```sql block, else from the first WITH/SELECT."""
    if not text:
        return None
    blocks = re.findall(
        r"```(?:sql|sqlite)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE
    )
    cand = blocks[-1] if blocks else text
    m = re.search(r"(WITH\b.*|SELECT\b.*)", cand, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip().rstrip(";").strip() + ";" if m else None


def run_sql(sql):
    """Execute read-only with a time limit; None on error or timeout."""
    conn = sqlite3.connect(f"file:{_db_path()}?mode=ro", uri=True)
    deadline = time.monotonic() + _SQL_TIMEOUT
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        return [list(r) for r in conn.execute(sql).fetchall()]
    except (sqlite3.Error, ValueError):
        return None
    finally:
        conn.close()


def _norm(res):
    """Order-insensitive key tolerant to number formatting (separators, currency, units, '(N)')."""
    if res is None:
        return None
    out = []
    for row in res:
        r = []
        for v in row:
            m = isinstance(v, str) and re.fullmatch(
                r"(-?[\d,]*\d(?:\.\d+)?)\s*" + _UNIT, v.strip(), re.IGNORECASE
            )
            if isinstance(v, str) and re.fullmatch(
                r"[$£€]\s?-?[\d,]*\d(?:\.\d+)?", v.strip()
            ):
                v = v.strip()[1:].strip().replace(",", "")
            m2 = isinstance(v, str) and re.fullmatch(
                r"\((-?\d+(?:\.\d+)?)\)", v.strip()
            )
            if m2:
                v = m2.group(1)
            elif m:
                v = m.group(1)
            if isinstance(v, str) and re.fullmatch(
                r"-?\d{1,3}([, ]\d{3})+(\.\d+)?", v.strip()
            ):
                v = v.strip().replace(",", "").replace(" ", "")
            if isinstance(v, (int, float)) or (
                isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v.strip())
            ):
                f = float(v)
                r.append(round(f, 2) if not math.isnan(f) else None)
            else:
                r.append(str(v).strip())
        out.append(tuple(r))
    return sorted(out, key=str)


def results_match(gold, pred) -> bool:
    """Lenient execution match (row order, duplicates, number formats, extra columns)."""
    g, p = _norm(gold), _norm(pred)
    if p is None or g is None:
        return False
    if g == p:
        return True

    def strip(rs):
        return [
            tuple(re.sub(r"\s*\(\d+\)$", "", x) if isinstance(x, str) else x for x in r)
            for r in rs
        ]

    if _norm(strip(gold)) == _norm(strip(pred)):
        return True

    def alnum(v):
        return re.sub(r"[^0-9A-Za-zÀ-ÿĀ-ž]", "", str(v)).lower()

    two_part = (
        gold
        and pred
        and len(gold[0]) == 1
        and len(pred[0]) > 1
        and len(gold) == len(pred)
    )
    if two_part and sorted(alnum(r[0]) for r in gold) == sorted(
        alnum("".join(str(x) for x in r)) for r in pred
    ):
        return True
    if sorted(set(map(tuple, _norm(gold))), key=str) == sorted(
        set(map(tuple, _norm(pred))), key=str
    ):
        return True
    if (
        gold
        and pred
        and len(gold[0]) == 1
        and len(pred[0]) > 1
        and len(gold) == len(pred)
    ):
        return any(_norm([(row[j],) for row in pred]) == g for j in range(len(pred[0])))
    return False


def process_results(doc, results):
    gold = json.loads(doc["answer"])
    sql = extract_sql(results[0])
    ok = sql is not None and results_match(gold, run_sql(sql))
    return {"exec_acc": float(ok)}
