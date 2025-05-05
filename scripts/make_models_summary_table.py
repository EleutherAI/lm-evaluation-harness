import argparse
import json
from pprint import pprint
from statistics import mean, median

import pandas as pd
import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate W&B runs and compute RER & score averages."
    )

    parser.add_argument("--wandb_key", type=str, required=True, help="W&B API key")
    parser.add_argument(
        "--entity", type=str, default="slovak-nlp", help="W&B entity name"
    )
    parser.add_argument(
        "--project", type=str, default="sklep-test-nli", help="W&B project name"
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        choices=["mean", "median", "max"],
        default="mean",
        help="Aggregation method",
    )
    parser.add_argument(
        "--baseline", type=str, default="gerulata/slovakbert", help="Baseline model"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="./results.json",
        help="Path to output JSON file",
    )

    return parser.parse_args()


def get_agg_fn(name):
    return {
        "mean": mean,
        "median": median,
    }[name]


def fetch_results_wandb(entity, project, wandb_key):
    wandb.login(key=wandb_key)
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    task, model, f1, acc, pear = [], [], [], [], []
    for run in runs:
        _f1 = run.summary._json_dict.get("eval/f1", None)
        _acc = run.summary._json_dict.get("eval/accuracy", None)
        _pear = run.summary._json_dict.get("eval/pearson", None)
        f1.append(_f1)
        acc.append(_acc)
        pear.append(_pear)
        name = run.name.split("--")
        task.append(name[0])
        model.append(name[1])

    runs_df = pd.DataFrame(
        {"task": task, "model": model, "f1": f1, "acc": acc, "pear": pear}
    )
    return runs_df


def safe_agg(series, agg_method):
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return agg_method(cleaned)


def compute_rer_for_task(row, task_group, baseline):
    metric_col = next(
        (col for col in ["f1", "acc", "pear"] if not task_group[col].isna().all()), None
    )
    if metric_col is None or pd.isna(row[metric_col]):
        return None

    baseline_row = task_group[task_group["model"] == baseline]
    if baseline_row.empty or pd.isna(baseline_row.iloc[0][metric_col]):
        return None

    baseline_score = baseline_row.iloc[0][metric_col]
    return (
        (row[metric_col] - baseline_score) / (1 - baseline_score)
        if baseline_score != 1
        else 0.0
    )


def compute_rer(df, baseline):
    return df.assign(
        RER=df.apply(lambda row: compute_rer_for_task(row, df, baseline), axis=1)
    )


def compute_model_averages(df):
    model_averages = []

    for model, group in df.groupby("model"):
        avg_rer = group["RER"].dropna().mean()
        actual_scores = group.apply(
            lambda row: next(
                (row[col] for col in ["f1", "acc", "pear"] if pd.notna(row[col])), None
            ),
            axis=1,
        )
        avg_score = actual_scores.dropna().mean()

        model_averages.append(
            {"model": model, "avg_rer": avg_rer, "avg_score": avg_score}
        )

    return pd.DataFrame(model_averages)


def main():
    args = parse_args()
    agg_fn = get_agg_fn(args.agg_method)

    results = {}

    runs_df = fetch_results_wandb(args.entity, args.project, args.wandb_key)

    grouped_df = runs_df.groupby(["task", "model"], as_index=False).agg(
        {
            "f1": lambda x: safe_agg(x, agg_fn),
            "acc": lambda x: safe_agg(x, agg_fn),
            "pear": lambda x: safe_agg(x, agg_fn),
        }
    )

    grouped_with_rer = grouped_df.groupby("task", group_keys=False).apply(
        lambda df: compute_rer(df, args.baseline)
    )
    model_summary_df = compute_model_averages(grouped_with_rer)

    for _, row in model_summary_df.iterrows():
        model = row["model"]
        results[model] = {"AVG": {"score": row["avg_score"], "RER": row["avg_rer"]}}

    for _, row in grouped_with_rer.iterrows():
        model = row["model"]
        task = row["task"]
        rer = row["RER"]

        for metric in ["f1", "acc", "pear"]:
            if pd.notna(row[metric]):
                score_key = metric
                score_val = row[metric]
                break
        else:
            score_key = None
            score_val = None

        if model not in results:
            results[model] = {}

        results[model][task] = {score_key: score_val, "RER": rer}

    pprint(results)
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
