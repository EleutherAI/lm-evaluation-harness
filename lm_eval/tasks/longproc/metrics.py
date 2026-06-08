"""Evaluation metrics for the LongProc benchmark.

Ported from the original LongProc evaluators:
  https://github.com/princeton-pli/LongProc

Each ``process_results_*`` function receives a single *doc* (dict) and a
*results* list (whose first element is the model's generation string) and
returns a dict mapping metric names to scalar values.
"""

import contextlib
import hashlib
import logging
import os
import re
import string
import subprocess
import tempfile
from subprocess import PIPE, Popen


eval_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extract_with_tag(response: str, tag: str):
    """Extract content between <tag>...</tag>."""
    start = response.find(f"<{tag}>")
    end = response.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return response[start + len(tag) + 2 : end].strip()


# ---------------------------------------------------------------------------
# Path Traversal
# ---------------------------------------------------------------------------


def process_results_path_traversal(doc, results):
    prediction = results[0]
    gt = doc["reference_output"].strip()
    parsed_pred = _extract_with_tag(prediction, "Route")

    if parsed_pred is None:
        return {
            "score": 0.0,
            "accuracy": 0.0,
            "partial_accuracy": 0.0,
            "extraction_rate": 0.0,
        }

    parsed_pred = parsed_pred.strip()
    if gt == parsed_pred:
        return {
            "score": 1.0,
            "accuracy": 1.0,
            "partial_accuracy": 1.0,
            "extraction_rate": 1.0,
        }

    gt_lines = gt.split("\n")
    pred_lines = parsed_pred.split("\n")

    match_count = 0
    for gl, pl in zip(gt_lines, pred_lines, strict=False):
        if gl != pl:
            break
        match_count += 1

    partial = (match_count + 1) / len(gt_lines) if match_count < len(gt_lines) else 1.0
    return {
        "score": 0.0,
        "accuracy": 0.0,
        "partial_accuracy": partial,
        "extraction_rate": 1.0,
    }


# ---------------------------------------------------------------------------
# Theory-of-Mind Tracking
# ---------------------------------------------------------------------------

_TOM_STOPWORDS_RE = re.compile(
    r"\b(a|an|the|on|in|at|the|step|thinks|think|believes|believe|is|are|of|location|know|knows|belief)\b"
)


def _normalize_tom(s):
    """Lowercase, remove punctuation, articles/stopwords, extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation and ch != "\u2019")
    s = _TOM_STOPWORDS_RE.sub(" ", s)
    return " ".join(s.split())


def _extract_belief_content(line):
    if line.startswith("-"):
        belief = line.split("-", 1)[1].strip()
        return _normalize_tom(belief)
    return None


def process_results_tom_tracking(doc, results):
    prediction = results[0]
    gt_solution = doc["reference_output"]

    parsed_pred = "\n".join(
        line for line in prediction.splitlines() if line.strip().startswith("-")
    )
    parsed_gt = "\n".join(
        line for line in gt_solution.splitlines() if line.strip().startswith("-")
    )

    model_responses = parsed_pred.strip().split("\n")
    gt_responses = parsed_gt.strip().split("\n")

    model_beliefs = [
        _extract_belief_content(l)
        for l in model_responses
        if _extract_belief_content(l)
    ]
    gt_beliefs = [
        _extract_belief_content(l) for l in gt_responses if _extract_belief_content(l)
    ]

    if len(model_beliefs) == len(gt_beliefs) and all(
        a == b for a, b in zip(model_beliefs, gt_beliefs, strict=False)
    ):
        strict_accuracy = 1.0
        partial_accuracy = 1.0
    else:
        strict_accuracy = 0.0
        first_diff = next(
            (
                i
                for i, (a, b) in enumerate(zip(model_beliefs, gt_beliefs, strict=False))
                if a != b
            ),
            None,
        )
        if first_diff is not None:
            partial_accuracy = first_diff / len(gt_beliefs) if gt_beliefs else 0.0
        else:
            partial_accuracy = (
                min(len(model_beliefs), len(gt_beliefs)) / len(gt_beliefs)
                if gt_beliefs
                else 0.0
            )

    return {
        "score": strict_accuracy,
        "accuracy": strict_accuracy,
        "partial_accuracy": partial_accuracy,
        "extraction_rate": 1.0,
    }


# ---------------------------------------------------------------------------
# Countdown
# ---------------------------------------------------------------------------


def _evaluate_countdown_final_solution(nums, target, solution_text):
    """Check whether *solution_text* (3 lines of equations) is valid."""
    nums = list(nums)

    def _parse_line(line):
        line = line.strip()
        parts = line.split("=")
        if len(parts) != 2:
            return False, None, None, None, None
        lhs, rhs = parts
        lhs_result = eval(lhs)  # noqa: S307 – trusted data
        if "+" in lhs:
            op = "+"
        elif "-" in lhs:
            op = "-"
        elif "*" in lhs:
            op = "*"
        elif "/" in lhs:
            op = "/"
        else:
            return False, None, None, None, None
        a, b = lhs.split(op)
        return lhs_result == int(rhs), int(a), int(b), int(rhs), op

    lines = solution_text.split("\n")
    if len(lines) != 3:
        return False
    for line in lines:
        try:
            correct, a, b, c, op = _parse_line(line)
        except (ValueError, SyntaxError):
            return False
        if not correct:
            return False
        if a not in nums:
            return False
        nums.remove(a)
        if b not in nums:
            return False
        nums.remove(b)
        nums.append(c)
    return nums[0] == target


def _evaluate_countdown_search_procedure(nums, target, procedure, gt_procedure):
    """Return (partial_accuracy, error_report)."""
    pred_lines = procedure.strip().split("\n")
    gt_lines = gt_procedure.strip().split("\n")

    if pred_lines[0] != gt_lines[0]:
        return 0.0, {
            "line_number": 0,
            "prediction": pred_lines[0],
            "ground_truth": gt_lines[0],
        }

    pred_lines = pred_lines[1:]
    gt_lines = gt_lines[1:]

    idx = -1
    error_report = {}
    for pred_l, gt_l in zip(pred_lines, gt_lines, strict=False):
        idx += 1
        if pred_l == gt_l:
            continue
        if "Pick two numbers" in gt_l:
            if pred_l != gt_l:
                error_report = {"line": idx, "pr": pred_l, "gt": gt_l}
                break
        elif "|- Try" in gt_l:
            pred_eq = pred_l.split("=")[0]
            gt_eq = gt_l.split("=")[0]
            if pred_eq != gt_eq:
                error_report = {"line": idx, "pr": pred_l, "gt": gt_l}
                break
            if ("drop this branch" in gt_l) != ("drop this branch" in pred_l):
                error_report = {"line": idx, "pr": pred_l, "gt": gt_l}
                break
        else:
            break

    return idx / len(gt_lines) if gt_lines else 0.0, error_report


def process_results_countdown(doc, results):
    prediction = results[0]
    metadata = doc["metadata_parsed"]
    nums = metadata["nums"]
    target = metadata["target"]

    pred_solution = _extract_with_tag(prediction, "Solution")

    if pred_solution is not None and _evaluate_countdown_final_solution(
        nums, target, pred_solution
    ):
        return {
            "score": 1.0,
            "accuracy": 1.0,
            "partial_accuracy": 1.0,
            "extraction_rate": 1.0,
        }

    extraction_rate = 1.0 if pred_solution is not None else 0.0

    if "# Search Procedure" not in prediction:
        return {
            "score": 0.0,
            "accuracy": 0.0,
            "partial_accuracy": 0.0,
            "extraction_rate": extraction_rate,
        }

    pred_procedure = prediction.split("# Search Procedure")[-1].strip()

    gt_procedure_full = doc["reference_output"]
    gt_procedure = (
        gt_procedure_full.split("# Search Procedure")[-1]
        .split("Now we have found the target")[0]
        .strip()
    )

    partial_accuracy, _ = _evaluate_countdown_search_procedure(
        nums, target, pred_procedure, gt_procedure
    )
    return {
        "score": 0.0,
        "accuracy": 0.0,
        "partial_accuracy": partial_accuracy,
        "extraction_rate": extraction_rate,
    }


# ---------------------------------------------------------------------------
# Travel Planning
# ---------------------------------------------------------------------------

_TRAVEL_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _parse_travel_response(response):
    """Parse a travel plan response into a list of (city, stay_days) tuples."""
    pattern_visit = r"\d+-\d+"
    pattern_flight = r".*Day (\d+).*from (\w+) to (\w+)"
    pattern_days = r"European cities for (\d+) days"

    days, flights, flight_days = [], [], []
    total_days = None
    for piece in response.split("\n"):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])

        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split("-")[1])
            if end_day == total_days:
                break
        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])

    visit_cities, parsed_plan = [], []
    for flight_day, begin_city, end_city in flights:
        flight_days.append(int(flight_day))
        if not visit_cities:
            visit_cities.append(begin_city)
            visit_cities.append(end_city)
        else:
            visit_cities.append(end_city)

    if not days or not flights or not visit_cities:
        return []

    last_day = int(days[-1].split("-")[1])
    flight_days = [1] + flight_days + [last_day]
    for i, visit_city in enumerate(visit_cities):
        city_stay = flight_days[i + 1] - flight_days[i] + 1
        parsed_plan.append((visit_city, city_stay))

    return parsed_plan


def _evaluate_travel_plan_solution(cities_str, durations_str, response):
    stays = [x for x in cities_str.split("**") if x]
    days = [int(x) for x in durations_str.split("**") if x]
    parsed_plan = _parse_travel_response(response)
    num_stays = min(len(stays), len(parsed_plan))
    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break
    return 0.0 if num_match / len(stays) < 1.0 else 1.0


def _normalize_travel_line(line):
    """Lower, remove punctuation, articles, extra whitespace."""

    def remove_articles(text):
        return _TRAVEL_ARTICLES_RE.sub(" ", text)

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    return " ".join(remove_articles(remove_punc(line.lower())).split())


def _evaluate_travel_search_procedure(output, gt_procedure):
    gt_procedure = gt_procedure.replace("Output the plan in the required format:", "")
    output = output.replace("Output the plan in the required format:", "")
    gt_procedure = "<Solving Procedure>" + gt_procedure.split("<Solving Procedure>")[1]
    if "<Solving Procedure>" in output:
        output = "<Solving Procedure>" + output.split("<Solving Procedure>")[1]

    pred_lines = [l.rstrip() for l in output.strip().split("\n") if l.strip()]
    gt_lines = [l.rstrip() for l in gt_procedure.strip().split("\n") if l.strip()]

    idx = -1
    for pred_l, gt_l in zip(pred_lines, gt_lines, strict=False):
        idx += 1
        if _normalize_travel_line(gt_l) in _normalize_travel_line(pred_l):
            continue
        else:
            break

    if idx < 0:
        idx = 0

    return idx / len(gt_lines) if gt_lines else 0.0


def process_results_travel_planning(doc, results):
    prediction = results[0]
    metadata = doc["metadata_parsed"]

    plan_text = _extract_with_tag(prediction, "Plan")
    extraction_rate = 0.0 if plan_text is None else 1.0

    if plan_text is not None:
        plan_text = plan_text.strip()
        accuracy = _evaluate_travel_plan_solution(
            metadata["ground_truth_cities"],
            metadata["ground_truth_durations"],
            plan_text,
        )
    else:
        accuracy = 0.0

    if accuracy == 1.0:
        partial_accuracy = 1.0
    else:
        gt_procedure = doc["reference_output"]
        partial_accuracy = _evaluate_travel_search_procedure(prediction, gt_procedure)

    return {
        "score": accuracy,
        "accuracy": accuracy,
        "partial_accuracy": partial_accuracy,
        "extraction_rate": extraction_rate,
    }


# ---------------------------------------------------------------------------
# HTML to TSV
# ---------------------------------------------------------------------------

_HTML_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize_answer_html(s):
    """Lower text, remove punctuation, articles, and all whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _HTML_ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s.replace(" ", "")


def _string_to_pd(text_lines):
    """Convert a list of TSV-formatted lines to a pandas DataFrame."""
    try:
        import csv

        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for html_to_tsv evaluation. "
            "Install it with: pip install pandas"
        ) from e

    try:
        reader = csv.reader(text_lines, delimiter="\t")
        data = list(reader)
        header = data[0]
        data = data[1:]
        if data and len(data[-1]) != len(header):
            data = data[:-1]
        for idx, line in enumerate(data):
            if len(line) != len(header):
                data[idx] = ["N/A"] * len(header)
        df = pd.DataFrame(data, columns=header).fillna("N/A")
        for col in df.columns:
            df[col] = df[col].astype(str).apply(_normalize_answer_html)
        return df
    except Exception:
        return None


def _compute_html_to_tsv_metrics(prediction_text, groundtruth_text):
    """Compute precision / recall / F1 between predicted and ground-truth TSV."""
    try:
        import pandas  # noqa: F401

        del pandas
    except ImportError as e:
        raise ImportError(
            "pandas is required for html_to_tsv evaluation. "
            "Install it with: pip install pandas"
        ) from e

    try:
        gt_lines = groundtruth_text.strip().split("\n")
        gt_df = _string_to_pd(gt_lines)
        pred_lines = prediction_text.strip().split("\n")
        pred_df = _string_to_pd(pred_lines)

        precision = 0.0
        for i in range(len(pred_df.index)):
            corr = gt_df.eq(pred_df.iloc[i].values).all(axis=1).any()
            precision += corr
        precision /= len(pred_df.index)

        recall = 0.0
        for i in range(len(gt_df.index)):
            corr = pred_df.eq(gt_df.iloc[i].values).all(axis=1).any()
            recall += corr
        recall /= len(gt_df.index)

        f1 = (
            2 * recall * precision / (recall + precision)
            if (recall + precision)
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1, "error": None}
    except Exception as exc:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(exc)}


def process_results_html_to_tsv(doc, results):
    prediction = results[0]

    try:
        search = re.search(r"```tsv([\s\S]*)```", prediction)
        if search is not None:
            prediction = search.group(1).strip()
        else:
            if "```tsv" in prediction:
                prediction = prediction.split("```tsv")[1]
            prediction = prediction.strip()
    except Exception:
        return {
            "score": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "extraction_rate": 0.0,
        }

    gt = doc["reference_output"]
    m = _compute_html_to_tsv_metrics(prediction, gt)

    extraction_rate = 1.0 if m["error"] is None else 0.0
    return {
        "score": m["f1"],
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "extraction_rate": extraction_rate,
    }


# ---------------------------------------------------------------------------
# Pseudocode to Code
# ---------------------------------------------------------------------------

_SPOC_IMPORT_HEADER = """\
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <list>
#include <fstream>
#include <climits>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <bitset>
using namespace std;
"""


def _evaluate_spoc_code(code, testcases):
    """Compile and execute C++ *code* against *testcases*.

    Uses a temporary directory for isolation.  Returns (success: bool, error_msg: str).
    """
    full_program = _SPOC_IMPORT_HEADER + code
    file_hash = hashlib.sha1(full_program.encode("utf-8")).hexdigest()[-16:]

    tmpdir = tempfile.mkdtemp(prefix="longproc_spoc_")
    cpp_file = os.path.join(tmpdir, file_hash + ".cpp")
    exe_file = os.path.join(tmpdir, file_hash + ".bin")

    try:
        with open(cpp_file, "w") as f:
            f.write(full_program)

        # Compile
        try:
            subprocess.check_output(
                ["g++", "-std=c++11", "-O", "-o", exe_file, cpp_file],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            return False, "compilation error"
        except FileNotFoundError:
            eval_logger.warning(
                "g++ not found – pseudo_to_code evaluation requires g++ with C++11 support"
            )
            return False, "g++ not found"

        # Execute against test cases
        for inputs, outputs in testcases[:10]:
            proc = Popen([exe_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            msg = ("\n".join(inputs) + "\n").encode("utf-8")
            try:
                proc_out, _ = proc.communicate(msg, timeout=3)
                proc_out = proc_out.decode("utf-8")
                expected_out = "\n".join(outputs) + "\n"
                if proc_out != expected_out:
                    return False, "inconsistent"
            except subprocess.TimeoutExpired:
                proc.kill()
                return False, "timeout"
            except subprocess.CalledProcessError:
                return False, "exec error"
            except UnicodeDecodeError:
                return False, "decode error"
            except Exception:
                return False, "unknown error"

        return True, ""
    finally:
        # Clean up temp files
        for f in (cpp_file, exe_file):
            with contextlib.suppress(OSError):
                os.remove(f)
        with contextlib.suppress(OSError):
            os.rmdir(tmpdir)


def process_results_pseudo_to_code(doc, results):
    prediction = results[0]
    metadata = doc["metadata_parsed"]

    try:
        parsed_pred = re.sub(r"```c\+\+", "```cpp", prediction)
        parsed_pred = re.search(r"```cpp([\s\S]*)```", parsed_pred).group(1)
    except Exception:
        return {
            "score": 0.0,
            "accuracy": 0.0,
            "partial_accuracy": 0.0,
            "extraction_rate": 0.0,
        }

    testcases = metadata["testcases"]
    success, _ = _evaluate_spoc_code(parsed_pred, testcases)

    if success:
        return {
            "score": 1.0,
            "accuracy": 1.0,
            "partial_accuracy": 1.0,
            "extraction_rate": 1.0,
        }
    else:
        return {
            "score": 0.0,
            "accuracy": 0.0,
            "partial_accuracy": 0.0,
            "extraction_rate": 1.0,
        }
