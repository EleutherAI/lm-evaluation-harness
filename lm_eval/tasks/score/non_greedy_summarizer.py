# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import List

import pandas as pd

from lm_eval.tasks.score.math.math_grader import math_equal
from lm_eval.utils import handle_non_serializable, make_table


N_SEEDS = 5


def load_json_logs(file_paths, subtasks):
    """
    Loads JSON logs of jsonl format from file paths into a single DataFrame.

    Args:
        file_paths: List of file paths to the JSON logs.

    Returns:
        A DataFrame containing the logs.
    """
    per_seed_df = {
        "question_id": [],
        "final_answer_seed_": [],
        "gt": [],
        "category": [],
    }
    _search_key = None
    for i in range(len(file_paths)):
        file_path = file_paths[i]
        with open(file_path, "r") as f:
            for line in f:
                datapoint = json.loads(line)
                if _search_key is None:
                    if "non_greedy_macro_accuracy" in datapoint:
                        _search_key = "non_greedy_macro_accuracy"
                    elif "non_greedy_accuracy" in datapoint:
                        _search_key = "non_greedy_accuracy"
                question_id, final_answer, gt, category = datapoint[_search_key]
                if subtasks is not None:
                    category = subtasks[i]
                per_seed_df["question_id"].append(question_id)
                per_seed_df["final_answer_seed_"].append(final_answer)
                per_seed_df["gt"].append(gt)
                per_seed_df["category"].append(category)
    df = pd.DataFrame(per_seed_df)
    return df


def calculate_consistency_rate(responses: List[List[str]]) -> float:
    """
    Calculate the Consistency Rate (CR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.

    Returns:
    The consistency rate as a float.
    """
    total_similarity = 0
    total_combinations = 0

    for response_set in responses:
        pairs = combinations(response_set, 2)
        num_pairs = len(response_set) * (len(response_set) - 1) / 2
        total_combinations += num_pairs
        for answer1, answer2 in pairs:
            total_similarity += int(answer1 == answer2)

    return total_similarity / total_combinations if total_combinations > 0 else 0.0


def calculate_math_consistency_rate(responses: List[List[str]]) -> float:
    """
    Calculate the Consistency Rate (CR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.

    Returns:
    The consistency rate as a float.
    """
    total_similarity = 0
    total_combinations = 0

    for response_set in responses:
        pairs = combinations(response_set, 2)
        num_pairs = len(response_set) * (len(response_set) - 1) / 2
        total_combinations += num_pairs
        for answer1, answer2 in pairs:
            total_similarity += int(math_equal(answer1, answer2))

    return total_similarity / total_combinations if total_combinations > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Calculate consistency rate from JSON logs."
    )
    parser.add_argument(
        "--log_dir", help="Path to the directory containing the JSON log files."
    )
    parser.add_argument("--dataset", help="Dataset name: agieval, mmlu_pro or math")
    args = parser.parse_args()

    for seed in range(1, N_SEEDS + 1):
        # Checking if directories exist
        seed_log_dir = os.path.join(args.log_dir, f"seed_{seed}")
        assert os.path.exists(
            seed_log_dir
        ), f"No logs found for seed={seed}. No directory found at {seed_log_dir}"
        subtasks = None
        if args.dataset == "agieval":
            agieval_subtasks = [
                "aqua_rat",
                "logiqa_en",
                "lsat_ar",
                "lsat_lr",
                "lsat_rc",
                "sat_en",
                "sat_math",
            ]
            subtasks = agieval_subtasks
            file_paths = []
            for subtask in agieval_subtasks:
                log_path = os.path.join(
                    seed_log_dir,
                    f"*/samples_non_greedy_robustness_agieval_{subtask}_*.jsonl",
                )
                subtask_logs = glob.glob(log_path)
                if len(subtask_logs) == 0:
                    raise FileNotFoundError(
                        f"No logs found for agieval subtask {subtask} for seed={seed} in the path {log_path}."
                    )
                elif len(subtask_logs) > 1:
                    raise FileExistsError(
                        f"Multiple logs found for agieval subtask {subtask} for seed={seed}."
                    )
                file_paths.append(subtask_logs[0])

        elif args.dataset == "mmlu_pro":
            task_logs = glob.glob(
                os.path.join(
                    seed_log_dir,
                    "*/samples_score_non_greedy_robustness_mmlu_pro_*.jsonl",
                )
            )
            file_paths = []
            if len(task_logs) == 0:
                raise FileNotFoundError(
                    f"No logs found for mmlu_pro for seed={seed}. PATH: {seed_log_dir}"
                )
            elif len(task_logs) > 1:
                raise FileExistsError(
                    f"Multiple logs found for mmlu_pro for seed={seed}."
                )
            file_paths.append(task_logs[0])

        elif args.dataset == "math":
            math_subtasks = [
                "algebra",
                "counting_and_prob",
                "geometry",
                "intermediate_algebra",
                "num_theory",
                "prealgebra",
                "precalc",
            ]
            subtasks = math_subtasks
            file_paths = []

            for subtask in math_subtasks:
                log_path = os.path.join(
                    seed_log_dir,
                    f"*/samples_non_greedy_robustness_math_{subtask}_*.jsonl",
                )

                subtask_logs = glob.glob(log_path)
                if len(subtask_logs) == 0:
                    raise FileNotFoundError(
                        f"No logs found for math subtask {subtask} for seed={seed} in the path {log_path}."
                    )
                elif len(subtask_logs) > 1:
                    raise FileExistsError(
                        f"Multiple logs found for math subtask {subtask} for seed={seed}."
                    )
                file_paths.append(subtask_logs[0])

        else:
            raise ValueError(
                "Invalid dataset name. only agieval, mmlu_pro and math are supported."
            )

        df = load_json_logs(file_paths, subtasks)

        # merge all dfs by question_id, category and gt
        if seed == 1:
            df_all = df
            df_all[f"final_answer_seed_{seed}"] = df["final_answer_seed_"]
        else:
            df_all = df_all.merge(
                df, on=["question_id", "category"], suffixes=("", seed)
            )

    responses = df_all[
        [f"final_answer_seed_{seed}" for seed in range(1, N_SEEDS + 1)]
    ].values.tolist()

    # calculate per seed accuracy

    if args.dataset == "math":
        consistency_rate = calculate_math_consistency_rate(responses)
        results = {"alias": f"score_non_greedy_robustness_{args.dataset}"}

        results.update(
            {
                "consistency_rate,none": consistency_rate,
                "consistency_rate_stderr,none": "N/A",
            }
        )

        for seed in range(1, N_SEEDS + 1):
            df_all[f"accuracy_seed_{seed}"] = df_all[
                [f"final_answer_seed_{seed}", "gt"]
            ].apply(lambda x: math_equal(*x), axis=1)
            accuracy = df_all[f"accuracy_seed_{seed}"].mean()
            results[f"seed_{seed}_accuracy,none"] = accuracy
            results[f"seed_{seed}_accuracy_stderr,none"] = "N/A"

    else:
        consistency_rate = calculate_consistency_rate(responses)
        results = {"alias": f"score_non_greedy_robustness_{args.dataset}"}

        results.update(
            {
                "consistency_rate,none": consistency_rate,
                "consistency_rate_stderr,none": "N/A",
            }
        )

        for seed in range(1, N_SEEDS + 1):
            df_all[f"accuracy_seed_{seed}"] = (
                df_all[f"final_answer_seed_{seed}"] == df_all["gt"]
            )
            accuracy = df_all[f"accuracy_seed_{seed}"].mean()
            results[f"seed_{seed}_accuracy,none"] = accuracy
            results[f"seed_{seed}_accuracy_stderr,none"] = "N/A"

    metrics = [f"seed_{seed}_accuracy" for seed in range(1, N_SEEDS + 1)] + [
        "consistency_rate"
    ]
    higher_is_better = {metric: True for metric in metrics}

    results_dict = {
        "results": {f"score_non_greedy_robustness_{args.dataset}": results},
        "group_subtasks": {f"score_non_greedy_robustness_{args.dataset}": []},
        "configs": None,
        "versions": {f"score_non_greedy_robustness_{args.dataset}": 1},
        "n-shot": {f"score_non_greedy_robustness_{args.dataset}": 0},
        "higher_is_better": {
            f"score_non_greedy_robustness_{args.dataset}": higher_is_better
        },
        "n-samples": None,
    }

    dumped = json.dumps(
        results_dict,
        indent=2,
        default=handle_non_serializable,
        ensure_ascii=False,
    )

    path = Path(args.log_dir)
    path.mkdir(parents=True, exist_ok=True)

    date_id = datetime.now().isoformat().replace(":", "-")
    file_results_aggregated = path.joinpath(f"{args.dataset}_results_{date_id}.json")
    file_results_aggregated.open("w", encoding="utf-8").write(dumped)

    print(make_table(results_dict))


if __name__ == "__main__":
    main()
