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
import wandb
import logging
from datetime import datetime
import wandb.apis.reports as wr
from lm_eval import utils


logger = logging.getLogger(__name__)

api = wandb.Api()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide a list of wandb run ids or report URLs to generate a report comparing different evaluation runs."
    )
    parser.add_argument(
        'runs', 
        metavar='STRING',
        type=lambda s: s.split(','),
        help='A list of W&B run_paths or report URLs separated by commas'
    )
    parser.add_argument(
        "--wandb_project",
        help="The project where you want to log the comparison report.",
    )
    parser.add_argument(
        "--wandb_entity",
        help="The wandb entity.", # TODO: better desp
    )
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
                    uncommon_items_with_indices[item].append(run_path[i])

    return common_items, uncommon_items_with_indices
    

def main():
    args = parse_args()
    run_paths = args.runs
    print("Comparing evaluation results with W&B run ids: ", run_paths)
    assert type(run_paths) == list

    eval_runs = {}
    for run_path in run_paths:
        assert len(run_path.split("/")) == 3, """
        The run_path is expected to be of the format <entity>/<project name>/<run id> and not {}
        """.format(run_path)

        eval_runs[run_path] = {}

        run = api.run(f"{run_path}")

        eval_runs[run_path]["url"] = run.url

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
                    if type(value) == float or type(value) == int:
                        eval_metrics[summary_key] = value

        eval_runs[run_path]["eval_metrics"] = eval_metrics

    print(eval_runs)

    all_tasks = [value["tasks"] for value in eval_runs.values()]
    print("All tasks: ", all_tasks)

    common_tasks, uncommon_tasks_with_run_ids = find_common_and_uncommon_items(all_tasks, run_paths)
    print(common_tasks)
    print(uncommon_tasks_with_run_ids)

    all_eval_metrics = [value["eval_metrics"] for value in eval_runs.values()]
    print(all_eval_metrics)

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
    print(wandb_project, wandb_entity)
    report = wr.Report(
        project=wandb_project,
        entity=wandb_entity,
        title=f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Evaluation Comparison Report",
        description=f"Comparing evaluations - run by: {wandb_entity}",
    )
    print(report)

    run_urls = [value["url"] for value in eval_runs.values()]
    run_urls_str =  "\n" + "* " + "\n* ".join(run_urls) + "\n"
    tasks_str =  "\n" + "* " + "\n* ".join(sorted(common_tasks)) + "\n"

    blocks = (
        [
            wr.TableOfContents(),
            wr.MarkdownBlock(
                "This report is comparing the evaluation results from the following evaluation runs: "
                f"{run_urls_str}"
            ),
            wr.H2("Comparing Evaluation Results by Common Tasks")
            wr.MarkdownBlock(
                "The following tasks are common in the provided evaluation runs: "
                f"{tasks_str}"
            ),
            
            # wr.H1("Complete Evaluation Results"),
            # wr.WeaveBlockSummaryTable(
            #     project=self.run.project,
            #     entity=self.run.entity,
            #     table_name="evaluation/eval_results",
            # ),
            # wr.PanelGrid(
            #     runsets=[
            #         wr.Runset(
            #             project=self.run.project,
            #             entity=self.run.entity,
            #         ).set_filters_with_python_expr(
            #             f'Name == "{str(self.run.name)}"'
            #         ),
            #     ]
            # ),
            wr.H1("Evaluation Results By Task"),
        ]
    )

    report.blocks = blocks
    report.save()
    wandb.termlog(f"📝 Check out the autogenerated report at: {report.url}")

    
# def prepare_report_by_task(tasks, results):
#     blocks = []
#     for task_name in self.task_names:
#         blocks.append(wr.H2(task_name))
#         panels = []
#         for metric_name, metric_value in results.items():
#             if task_name in metric_name:
#                 panels.append(
#                     wr.ScalarChart(
#                         title=f"{metric_name}",
#                         metric=f"{metric_name}",
#                         font_size="large",
#                     )
#                 )
#         _results = {
#             "results": {f"{task_name}": self.results.get("results").get(task_name)},
#             "versions": {
#                 f"{task_name}": self.results.get("versions").get(task_name)
#             },
#             "n-shot": {f"{task_name}": self.results.get("n-shot").get(task_name)},
#         }
#         results_md = utils.make_table(_results)
#         blocks.extend([wr.MarkdownBlock(results_md), wr.PanelGrid(panels=panels)])
#         # TODO: Add results table

#     return blocks
    

if __name__ == "__main__":
    main()