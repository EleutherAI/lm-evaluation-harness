"""
Smoke-test for the zero_scrolls lm-eval task implementation.

Tests:
  1. Task registration  — all 10 tasks + the group appear in TaskManager
  2. process_docs       — query/context extraction from index fields
  3. Metrics            — rouge_geomean, f1, accuracy, es, concordance
  4. End-to-end         — synthetic dataset through the full lm-eval pipeline

Run with:
    python3 test_zero_scrolls.py
"""

import sys
import textwrap
from pathlib import Path

# ── make sure we're running from the repo root ──────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

errors = []


def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f"  →  {detail}" if detail else ""))
        errors.append(label)


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}1. Task registration{RESET}")
# ─────────────────────────────────────────────────────────────────────────────
from lm_eval.tasks import TaskManager

tm = TaskManager()
all_zs = sorted(t for t in tm.all_tasks if "zero_scroll" in t)
print(f"  Found: {all_zs}")

expected_tasks = [
    "zero_scrolls_book_sum_sort",
    "zero_scrolls_govreport",
    "zero_scrolls_musique",
    "zero_scrolls_narrative_qa",
    "zero_scrolls_qasper",
    "zero_scrolls_qmsum",
    "zero_scrolls_quality",
    "zero_scrolls_space_digest",
    "zero_scrolls_squality",
    "zero_scrolls_summ_screen_fd",
]
for t in expected_tasks:
    check(t, t in all_zs)
check("zero_scrolls group", "zero_scrolls" in all_zs)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}2. process_docs — index field extraction{RESET}")
# ─────────────────────────────────────────────────────────────────────────────
import importlib.util

spec = importlib.util.spec_from_file_location(
    "utils", ROOT / "lm_eval/tasks/zero_scrolls/utils.py"
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

from datasets import Dataset

sample_qa = {
    "input": "Who wrote Hamlet?\n\nShakespeare wrote the play Hamlet around 1600.",
    "output": "Shakespeare",
    "id": "1",
    "pid": "1",
    "query_start_index": 0,
    "query_end_index": 17,          # "Who wrote Hamlet?"
    "document_start_index": 19,
    "document_end_index": 63,
    "truncation_separator": "",
    "inner_docs_start_indices": [],
}
sample_summ = {
    **sample_qa,
    "query_start_index": -1,
    "query_end_index": -1,
    "document_start_index": 0,
    "document_end_index": 63,
}

ds = Dataset.from_list([sample_qa, sample_summ])
processed = utils.process_docs(ds)

check(
    "QA: question extracted",
    processed[0]["question"] == "Who wrote Hamlet?",
    repr(processed[0]["question"]),
)
check(
    "QA: context extracted",
    "Shakespeare" in processed[0]["context"],
    repr(processed[0]["context"]),
)
check(
    "Summarisation: question is empty string",
    processed[1]["question"] == "",
    repr(processed[1]["question"]),
)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}3. Metric functions{RESET}")
# ─────────────────────────────────────────────────────────────────────────────

# ROUGE geomean
doc_rouge = {"output": "Shakespeare wrote Hamlet around 1600."}
r = utils.process_rouge(doc_rouge, ["Shakespeare wrote the play Hamlet."])
check("process_rouge returns rouge_geomean", "rouge_geomean" in r)
check("process_rouge score > 0", r.get("rouge_geomean", 0) > 0, str(r))

r_empty = utils.process_rouge({"output": ""}, ["some reference"])
check("process_rouge handles empty prediction → 0", r_empty["rouge_geomean"] == 0.0)

# Token F1
doc_qa = {"output": "Shakespeare"}
f = utils.process_qa_f1(doc_qa, ["shakespeare"])
check("process_qa_f1 perfect match → 100", abs(f["f1"] - 100.0) < 1e-6, str(f))

f2 = utils.process_qa_f1(doc_qa, ["Marlowe"])
check("process_qa_f1 no-match → 0", f2["f1"] == 0.0, str(f2))

f3 = utils.process_qa_f1({"output": None}, ["anything"])
check("process_qa_f1 handles None output → 0", f3["f1"] == 0.0)

# Accuracy (quality)
a = utils.process_accuracy({"output": "B"}, ["(B) Hamlet"])
check("process_accuracy correct letter → 1.0", a["acc"] == 1.0, str(a))

a2 = utils.process_accuracy({"output": "A"}, ["(B) Hamlet"])
check("process_accuracy wrong letter → 0.0", a2["acc"] == 0.0, str(a2))

# Exponential similarity (space_digest)
e = utils.process_space_digest({"output": "75"}, ["75"])
check("process_space_digest perfect → 100", abs(e["es"] - 100.0) < 1e-6, str(e))

e2 = utils.process_space_digest({"output": "25"}, ["75"])
# 50-pt deviation: ES = 2^(-10 * 0.5) = 2^-5 = 3.125
check("process_space_digest 50-pt error → ~3.125", abs(e2["es"] - 3.125) < 0.01, str(e2))

e3 = utils.process_space_digest({"output": "no number here"}, ["75"])
check("process_space_digest unparseable → 0", e3["es"] == 0.0)

# Concordance index (book_sum_sort)
c = utils.process_book_sum_sort({"output": "1, 2, 3"}, ["1, 2, 3"])
check("process_book_sum_sort perfect order → 100", abs(c["concordance"] - 100.0) < 1e-6, str(c))

c2 = utils.process_book_sum_sort({"output": "3, 2, 1"}, ["1, 2, 3"])
check("process_book_sum_sort reversed → 0", c2["concordance"] == 0.0, str(c2))

c3 = utils.process_book_sum_sort({"output": "1, 3, 2"}, ["1, 2, 3"])
check(
    "process_book_sum_sort one swap → 66.7",
    abs(c3["concordance"] - 66.666) < 0.1,
    str(c3),
)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}4. End-to-end pipeline with synthetic dataset{RESET}")
# ─────────────────────────────────────────────────────────────────────────────
from unittest.mock import patch, MagicMock
import lm_eval

# Build a tiny synthetic dataset that looks exactly like tau/zero_scrolls govreport
synthetic_rows = [
    {
        "input": "The government published a report on climate in 2023 showing record temperatures.",
        "output": "2023 climate report shows record temperatures.",
        "id": f"doc{i}",
        "pid": f"doc{i}",
        "query_start_index": -1,
        "query_end_index": -1,
        "document_start_index": 0,
        "document_end_index": 79,
        "truncation_separator": "",
        "inner_docs_start_indices": [],
    }
    for i in range(4)
]

from datasets import DatasetDict

synthetic_ds = DatasetDict({"validation": Dataset.from_list(synthetic_rows)})

with patch("datasets.load_dataset", return_value=synthetic_ds):
    results = lm_eval.simple_evaluate(
        model="dummy",
        tasks=["zero_scrolls_govreport"],
        limit=3,
        verbosity="WARNING",
    )

score = results["results"]["zero_scrolls_govreport"]["rouge_geomean,none"]
check(
    "End-to-end: rouge_geomean key present in results",
    "rouge_geomean,none" in results["results"]["zero_scrolls_govreport"],
)
check(
    "End-to-end: dummy model scores 0 (expected, outputs empty string)",
    score == 0.0,
    f"got {score}",
)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─' * 55}")
if errors:
    print(f"{FAIL} {len(errors)} check(s) failed: {errors}")
    sys.exit(1)
else:
    total = 11 + 3 + 14 + 2   # registration + process_docs + metrics + e2e
    print(f"{PASS} All checks passed ({total} assertions)")
