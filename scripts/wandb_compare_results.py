"""
inputs: run ids or reports urls
how: list of strings, local wandb dir, {--wandb_group, --tag} mongodb style

if run ids: do something
if reports urls: parse the run ids from the description
if run ids not available: gracefully fail with a error log

run ids: list

metrics = []
for run_id in run_ids:
    run = get_run(run_id)
    inspect run and get metrics
    metrics.append(metric)

assert same metrics
find all the similar metrics and keep a tab of missing metrics.

get the configs too for run comparator

create a new report:
    - same structure as the solo report
    - bar charts
    - run comparator


test with: ayush-thakur/lm-eval-harness-integration/cw53qh3d,ayush-thakur/lm-eval-harness-integration/k75giqms
"""

import argparse
import logging
from datetime import datetime

import wandb
import wandb.apis.reports as wr


logger = logging.getLogger(__name__)

api = wandb.Api()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide a list of wandb run ids or report URLs to generate a report comparing different evaluation runs."
    )
    parser.add_argument(
        "runs",
        metavar="STRING",
        type=lambda s: s.split(","),
        help="A list of W&B run_paths or report URLs separated by commas",
    )
    parser.add_argument(
        "--wandb_project",
        help="The project where you want to log the comparison report.",
    )
    parser.add_argument(
        "--wandb_entity",
        help="The wandb entity.",  # TODO: better desp
    )
    # TODO: take group and tag as input
    return parser.parse_args()


def find_common_and_uncommon_items(tasks, run_paths):
    # Find common items
    common_items = set(tasks[0]).intersection(*tasks[1:])

    # Find uncommon items with their indices
    uncommon_items_with_indices = {}
    for i, lst in enumerate(tasks):
        for item in lst:
            if item not in common_items:
                if item not in uncommon_items_with_indices:
                    uncommon_items_with_indices[item] = [run_paths[i]]
                else:
                    uncommon_items_with_indices[item].append(run_paths[i])

    return common_items, uncommon_items_with_indices


def get_metrics_per_task(data, tasks):
    metrics_per_task = {task: set() for task in tasks}

    for d in data:
        for key, value in d.items():
            task = key.split("/")[0]
            if task in tasks:
                metrics_per_task[task].add(key)

    return metrics_per_task


def prepare_report_by_task(tasks, run_ids, metrics_by_tasks):
    task_blocks = []
    for task in tasks:
        task_blocks.append(wr.H3(task))

        runsets = []
        for run_id in run_ids:
            runsets.append(
                wr.Runset(
                    project=wandb_project, entity=wandb_entity
                ).set_filters_with_python_expr(f"ID == {run_id}")
            )

        panels = []
        metrics = metrics_by_tasks[task]
        for metric in metrics:
            panels.append(wr.BarPlot(metrics=[metric]))

        task_blocks.append(
            wr.PanelGrid(
                runsets=runsets,
                panels=panels,
            )
        )

    return task_blocks


def main():
    args = parse_args()
    run_paths = args.runs
    print("Comparing evaluation results with W&B run ids: ", run_paths)
    assert isinstance(run_paths, list)

    eval_runs = {}
    run_ids = []
    for run_path in run_paths:
        assert len(run_path.split("/")) == 3, """
        The run_path is expected to be of the format <entity>/<project name>/<run id> and not {}
        """.format(run_path)

        eval_runs[run_path] = {}

        run = api.run(f"{run_path}")

        eval_runs[run_path]["url"] = run.url
        run_ids.append(run.id)

        config = run.config
        eval_runs[run_path]["config"] = config

        task_config = config.get("task_configs")
        tasks = list(task_config.keys())
        eval_runs[run_path]["tasks"] = tasks

        summary = run.summary
        summary_keys = list(summary.keys())
        eval_metrics = {}
        for task in tasks:
            for summary_key in summary_keys:
                if task in summary_key:
                    value = summary[summary_key]
                    if isinstance(value, float) or isinstance(value, int):
                        eval_metrics[summary_key] = value

        eval_runs[run_path]["eval_metrics"] = eval_metrics

    all_tasks = [value["tasks"] for value in eval_runs.values()]

    common_tasks, uncommon_tasks_with_run_ids = find_common_and_uncommon_items(
        all_tasks, run_paths
    )

    all_eval_metrics = [value["eval_metrics"] for value in eval_runs.values()]

    metrics_by_tasks = get_metrics_per_task(all_eval_metrics, common_tasks)

    global wandb_project, wandb_entity
    wandb_project = (
        args.wandb_project
        if args.wandb_project is not None
        else run_paths[0].split("/")[1]
    )
    wandb_entity = (
        args.wandb_entity
        if args.wandb_entity is not None
        else wandb.apis.PublicApi().default_entity
    )
    report = wr.Report(
        project=wandb_project,
        entity=wandb_entity,
        title=f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Evaluation Comparison Report",
        description=f"Comparing evaluations - run by: {wandb_entity}",
    )

    run_urls = [value["url"] for value in eval_runs.values()]
    run_urls_str = "\n" + "* " + "\n* ".join(run_urls) + "\n"
    tasks_str = "\n" + "* " + "\n* ".join(sorted(common_tasks)) + "\n"

    task_blocks = prepare_report_by_task(common_tasks, run_ids, metrics_by_tasks)

    # comparer_block = get_run_comparison_block(run_ids)

    blocks = (
        [
            wr.TableOfContents(),
            wr.MarkdownBlock(
                "This report is comparing the evaluation results from the following evaluation runs: "
                f"{run_urls_str}"
            ),
            wr.H2("Comparing Evaluation Results by Common Tasks"),
            wr.MarkdownBlock(
                "The following tasks are common in the provided evaluation runs: "
                f"{tasks_str}"
            ),
        ]
        + task_blocks
        + [
            wr.H2("Comparing Configurations"),
        ]
        # + comparer_block
        # TODO: Add wr.Gallery for each evaluation run.
    )

    report.blocks = blocks
    report.save()
    wandb.termlog(f"📝 Check out the autogenerated report at: {report.url}")


def get_run_comparison_block(run_ids):
    runsets = []
    for run_id in run_ids:
        runsets.append(
            wr.Runset(
                project=wandb_project, entity=wandb_entity
            ).set_filters_with_python_expr(f"ID == {run_id}")
        )

    panels = [wr.RunComparer(diff_only="split")]

    comparer_block = [wr.PanelGrid(runsets=runsets, panels=panels)]
    return comparer_block


if __name__ == "__main__":
    main()
