import argparse
import glob
import json
import os
from itertools import combinations
from typing import List

import pandas as pd

from lm_eval.tasks.score.math.math_grader import math_equal
from lm_eval.utils import make_table


def load_json_logs(file_paths, seed):
    """
    Loads JSON logs of jsonl format from file paths into a single DataFrame.

    Args:
        file_paths: List of file paths to the JSON logs.

    Returns:
        A DataFrame containing the logs.
    """
    per_seed_df = {
        "question_id": [],
        f"final_answer_seed_{seed}": [],
        "gt": [],
        "category": [],
    }
    for file_path in file_paths:
        with open(file_path, "r") as f:
            for line in f:
                datapoint = json.loads(line)
                question_id, final_answer, gt, category = datapoint[
                    "non_greedy_macro_accuracy"
                ]
                per_seed_df["question_id"].append(question_id)
                per_seed_df[f"final_answer_seed_{seed}"].append(final_answer)
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
    parser.add_argument("--dataset", help="Dataset name: agi_eval, mmlu_pro or math")
    args = parser.parse_args()

    for seed in range(1, 5):
        # Checking if directories exist
        seed_log_dir = os.path.join(args.log_dir, f"seed_{seed}")
        assert os.path.exists(
            seed_log_dir
        ), f"No logs found for seed={seed}. No directory found at {seed_log_dir}"

        if args.dataset == "agi_eval":
            agi_eval_subtasks = [
                "agieval_aqua_rat",
                "logiqa_en",
                "lsat_ar",
                "lsat_lr",
                "lsat_rc",
                "sat_en",
                "sat_math",
            ]
            file_paths = []
            for subtask in agi_eval_subtasks:
                subtask_logs = glob.glob(
                    os.path.join(
                        seed_log_dir,
                        f"seed_{seed}",
                        f"*/samples_score_non_greedy_robustness_agieval_{subtask}_*.jsonl",
                    )
                )
                if len(subtask_logs) == 0:
                    raise FileNotFoundError(
                        f"No logs found for agi_eval subtask {subtask} for seed={seed}."
                    )
                elif len(subtask_logs) > 1:
                    raise FileExistsError(
                        f"Multiple logs found for agi_eval subtask {subtask} for seed={seed}."
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
                "prompt_robustness_math_algebra",
                "counting_and_prob",
                "geometry",
                "intermediate_algebra",
                "num_theory",
                "prealgebra" "," "precalc",
            ]
            file_paths = []

            for subtask in math_subtasks:
                subtask_logs = glob.glob(
                    os.path.join(
                        seed_log_dir,
                        f"*/samples_score_non_greedy_robustness_math_{subtask}_*.jsonl",
                    )
                )
                if len(subtask_logs) == 0:
                    raise FileNotFoundError(
                        f"No logs found for math subtask {subtask} for seed={seed}."
                    )
                elif len(subtask_logs) > 1:
                    raise FileExistsError(
                        f"Multiple logs found for math subtask {subtask} for seed={seed}."
                    )
                file_paths.append(subtask_logs[0])

        else:
            raise ValueError(
                "Invalid dataset name. only agi_eval, mmlu_pro and math are supported."
            )

        df = load_json_logs(file_paths, seed)

        # merge all dfs by question_id, category and gt
        if seed == 1:
            df_all = df
        else:
            df_all = df_all.merge(df, on=["question_id", "category", "gt"])

    responses = df_all[
        [f"final_answer_seed_{seed}" for seed in range(1, 5)]
    ].values.tolist()

    # calculate per seed accuracy

    if args.dataset == "math":
        consistency_rate = calculate_math_consistency_rate(responses)
        results = {"consistency_rate": consistency_rate}
        for seed in range(1, 5):
            df_all[f"accuracy_seed_{seed}"] = df_all[
                [f"final_answer_seed_{seed}", "gt"]
            ].apply(lambda x: math_equal(*x), axis=1)
            accuracy = df_all[f"accuracy_seed_{seed}"].mean()
            results[f"seed_{seed}"] = accuracy

    else:
        consistency_rate = calculate_consistency_rate(responses)
        results = {"alias": f"score_non_greedy_robustness_{args.dataset}"}

        results.update(
            {
                "consistency_rate,none": consistency_rate,
                "consistency_rate_stderr,none": "N/A",
            }
        )

        for seed in range(1, 5):
            df_all[f"accuracy_seed_{seed}"] = (
                df_all[f"final_answer_seed_{seed}"] == df_all["gt"]
            )
            accuracy = df_all[f"accuracy_seed_{seed}"].mean()
            results[f"seed_{seed}_accuracy,none"] = accuracy
            results[f"seed_{seed}_accuracy_stderr,none"] = "N/A"

        metrics = [f"seed_{seed}_accuracy" for seed in range(1, 5)] + [
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

    print(make_table(results_dict))


if __name__ == "__main__":
    main()
