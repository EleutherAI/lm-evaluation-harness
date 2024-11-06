# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from functools import partial
from typing import Any, Dict, List

import datasets
import numpy as np

from lm_eval.tasks.score import utils
from lm_eval.tasks.score.math.math_grader import math_equal
from lm_eval.utils import eval_logger


TEMPLATE_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompt_templates.json")

PROMPT_ROBUSTNESS_TEMPLATE_KEY = "prompt_robustness"


def _remove_right_units(expr):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text" in expr:
        try:
            splits = re.split(r"\\text\s*{\s*", expr)
            # print(splits)
            assert len(splits) == 2 and splits[0] not in ("", "(")
            return splits[0]
        except AssertionError:
            pass

    if "\\text{" in expr:
        return re.sub(r"\\text{([^}]+)}", r"\1", expr)
    elif "\\mbox{" in expr:
        splits = expr.split("\\mbox{")
        assert len(splits) == 2
        return splits[0]
    else:
        return expr


def _process_and_or_inside_text(string):
    string = re.sub(r"\s*\\text{\s*(or|and)\s*}\s*", ",", string)
    string = re.sub(r",\s*,", ",", string)
    return string


def _remove_left_and_right(expr):
    """Remove the right and left latex commands."""
    expr = re.sub(r"\\left", "", expr)
    expr = re.sub(r"\\right", "", expr)
    return expr


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\s*\w+)", r"\\sqrt{\1}", string)
    return _string


def _fix_interval(expr):
    """Fix interval expression."""
    if "\\in " in expr:
        return expr.split("\\in ")[1].strip()

    return expr


def _fix_fracs(string):
    # replacing all extra spaces
    while "\\frac " in string:
        string = string.replace("\\frac ", "\\frac")
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    if "_" in x:
        # Due to base
        x = x.split("_")[0]
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  # implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _inject_implicit_mixed_fraction(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 \\frac{3}{4} => 7+3/4
    """
    p1 = re.compile(r"(\d+) *\\frac{(\d+)}{(\d+)}")

    def replacer(match):
        whole_part = match.group(1)
        numerator = match.group(2)
        denominator = match.group(3)

        if whole_part:
            return f"{whole_part} + {numerator}/{denominator}"
        else:
            return f"{numerator}/{denominator}"

    step = p1.sub(replacer, step)
    return step


def normalize_answer_string(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.

    expr = _remove_left_and_right(expr)
    expr = _process_and_or_inside_text(expr)
    expr = _remove_right_units(expr)
    expr = _fix_interval(expr)
    for surround_str in [
        "\\\\text",
        "\\\\mathrm",
        "\\\\mathcal",
        "\\\\textbf",
        "\\\\textit",
    ]:
        expr = expr.replace(surround_str, "")
        pattern = f"^{surround_str}" + "\{(?P<text>.+?)\}$"
        m = re.search(pattern, expr)
        if m is not None:
            expr = m.group("text")

    expr = expr.replace("\!", "")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace("^{\\circ}", "")

    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "p.m.",
        "PM",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)

    if "day" in expr:
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        weekday_expressed = False
        for day in days:
            if day in expr:
                weekday_expressed = True
                break

        if not weekday_expressed:
            expr = re.sub("day(s)?", "", expr)

    expr = re.sub("\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = _fix_sqrt(expr)

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    expr = _fix_fracs(expr)

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = _inject_implicit_mixed_fraction(expr)
    expr = expr.replace(" ", "")

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def find_boxed_entries(answer_str):
    stack = []
    results = []
    i = 0

    while i < len(answer_str):
        if answer_str[i : i + 7] == "\\boxed{":
            stack.append(i + 7)
            i += 7
        elif answer_str[i] == "{":
            if stack:
                stack.append(i + 1)
            i += 1
        elif answer_str[i] == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    results.append(answer_str[start:i])
            i += 1
        else:
            i += 1

    if len(results) == 0:
        raise ValueError("Not enough boxed entries")
    else:
        results = [normalize_answer_string(result) for result in results]

    if len(results) == 1:
        # Single boxed entry, trivial case
        return results

    else:
        # Multiple boxed entries. There are two cases possible
        # (a) The reference solution has the same question answered in multiple ways
        # (b) The answer is split across multiple boxed entries and we need to merge
        result_equal = True
        for idx in range(len(results) - 1):
            if not (results[idx] == results[idx + 1]):
                result_equal = False
                break

        if result_equal:
            # Same problem solved in multiple ways
            return [results[0]]
        else:
            return results


def extract_answer(solution: str, problem: str, corrected_answers: list) -> str:
    entries = find_boxed_entries(solution)

    if len(entries) == 1:
        parsed_answer = entries[0]

    if len(entries) > 1:
        for item in corrected_answers:
            if item["problem"] == problem:
                parsed_answer = item["answer"]
                break
        else:
            parsed_answer = ", ".join(entries)

    if not (
        ("Find the equation" in problem)
        or ("Enter the equation" in problem)
        or ("What is the equation" in problem)
        or ("described by the equation" in problem)
        or ("Find an equation" in problem)
    ) and ("=" in parsed_answer):
        if parsed_answer.count("=") == 1:
            # For greater count, it means we're just predicting values of multiple variables
            parsed_answer = parsed_answer.split("=")[1]
    return parsed_answer


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict, idx, corrected_answer) -> dict:
        out_doc = {
            "question": doc["problem"],
            "question_id": idx,
            "solution": doc["solution"],
            "answer": extract_answer(doc["solution"], doc["problem"], corrected_answer),
        }
        return out_doc

    corrected_answer_path = os.path.join(
        os.path.dirname(__file__), "to_be_fixed_questions.json"
    )

    with open(corrected_answer_path, "r") as f:
        corrected_answers = json.load(f)

    return dataset.map(
        partial(_process_doc, corrected_answer=corrected_answers), with_indices=True
    )


def prompt_robustness_process_docs(doc: datasets.Dataset) -> datasets.Dataset:
    doc = process_docs(doc)
    return utils.process_docs_add_prompts(
        doc,
        PROMPT_ROBUSTNESS_TEMPLATE_KEY,
        TEMPLATE_FILE_PATH,
    )


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]
    try:
        entries = find_boxed_entries(candidates)
    except ValueError:
        entries = []

    if len(entries) == 0:
        answer = None
    else:
        answer = entries[-1]

    if math_equal(answer, doc["answer"]):
        retval = 1
    else:
        retval = 0

    prompt_id = doc["prompt_id"]

    results = {
        f"{prompt_id}_accuracy": (prompt_id, retval),
    }
    return results


def per_prompt_accuracy(results: List[Dict[str, Any]], p_id=0) -> float:
    accuracies = []
    for result in results:
        prompt_id, retval = result
        if prompt_id != p_id:
            continue
        accuracies.append(retval)

    accuracy = sum(accuracies) / len(accuracies)
    eval_logger.info(f"Prompt - {prompt_id} accuracy: {accuracy}")

    return np.round(accuracy, 4)


per_prompt_accuracy_0 = partial(per_prompt_accuracy, p_id=0)
per_prompt_accuracy_1 = partial(per_prompt_accuracy, p_id=1)
per_prompt_accuracy_2 = partial(per_prompt_accuracy, p_id=2)
per_prompt_accuracy_3 = partial(per_prompt_accuracy, p_id=3)
per_prompt_accuracy_4 = partial(per_prompt_accuracy, p_id=4)
per_prompt_accuracy_5 = partial(per_prompt_accuracy, p_id=5)
per_prompt_accuracy_6 = partial(per_prompt_accuracy, p_id=6)
per_prompt_accuracy_7 = partial(per_prompt_accuracy, p_id=7)
per_prompt_accuracy_8 = partial(per_prompt_accuracy, p_id=8)
per_prompt_accuracy_9 = partial(per_prompt_accuracy, p_id=9)
