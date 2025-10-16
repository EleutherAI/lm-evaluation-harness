from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

from lm_eval.tasks.llmsql.utils import (
    _download_file,
    extract_table_name,
    find_sql,
    fix_table_name,
)


_evaluator = None  # Evaluator of the outputs


def get_evaluator() -> LLMQLEvaluator:
    """
    Returns:
        LLMQLEvaluator: used to not open connection for each call of process_results.
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = LLMQLEvaluator()
    return _evaluator


def process_results(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Takes the model output and evaluates with the help of .db file.

    Args:
        predictions (List[str]): output of the model.
        references (List[str]): reference sqls.

    Returns:
        Dict[str, float]:  the accuracy of the results of execution.
    """
    prediction = predictions[0]
    reference = references[0]
    evaluator = get_evaluator()
    acc = 0
    for prediction, reference in zip(predictions, references):
        acc += evaluator.evaluate_one(prediction, reference)
    return {"execution_accuracy": acc / len(references)}


def execute_sql(cursor, sql: str) -> list[tuple] | None:
    """
    Execute a SQL query on the given SQLite connection and return its results.

    The results are always sorted to avoid differences caused by row order (order agnostic).
    If the query fails, the function logs the error and returns None.

    Args:
        conn (sqlite3.Connection): An active SQLite database connection.
        sql (str): SQL query string to execute.

    Returns:
        Optional[List[Tuple]]:
            - Sorted list of result rows (each row as a tuple) if successful.
            - [(None,)] if the query executed but returned NULL values.
            - None if the SQL execution failed due to an exception.
    """
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        return sorted(results)
    except Exception:
        return None


class LLMQLEvaluator:
    """
    Persistent, reusable evaluator for SQL accuracy.
    Keeps a single SQLite connection open to avoid per-call overhead.
    """

    def __init__(self, db_path: str | None = None):
        workdir_path = Path("./llmsql_workdir")
        workdir_path.mkdir(parents=True, exist_ok=True)
        if db_path is None:
            db_path = workdir_path / "sqlite_tables.db"
            if not db_path.is_file():
                db_path = _download_file("sqlite_tables.db")
        self.conn = sqlite3.connect(
            Path(db_path).absolute(),
        )
        self.curr = self.conn.cursor()

    def evaluate_one(self, prediction: str, reference: str) -> float:
        """Evaluate a single prediction-reference pair."""
        try:
            gold_sql = reference
            table_id = extract_table_name(gold_sql)

            # If at least 1 produces SQL in the output is correct, the model passed the task.
            for sql in find_sql(prediction):
                fixed = fix_table_name(sql, table_id)
                res = execute_sql(self.curr, fixed)
                gold_res = execute_sql(self.curr, gold_sql)

                if gold_res == res:
                    return 1.0
            return 0.0
        except Exception:
            return 0.0
