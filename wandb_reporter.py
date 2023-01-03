import argparse
import logging

from packaging.version import Version
from lm_eval.utils import simple_parse_args_string
from lm_eval.tasks import TASK_REGISTRY
from lm_eval.models import get_model

logger = logging.getLogger(__name__)
WANDB_AVAILABLE = False
COMPARE_MODELS_VEGA_SPEC = "parambharat/evalharness-compare-models"

try:
    import wandb

    assert Version(wandb.__version__) >= Version("0.13.6")
    if Version(wandb.__version__) < Version("0.13.6"):
        wandb.require("report-editing:v0")
    WANDB_AVAILABLE = True
except:
    logger.warning(
        "To use the wandb reporting functionality please install wandb>=0.13.6.\n"
        "To install the latest version of wandb run `pip install wandb --upgrade`"
    )
    WANDB_AVAILABLE = False

if WANDB_AVAILABLE:
    from collections import defaultdict
    import json
    from datetime import datetime
    import inspect
    from pytablewriter import MarkdownTableWriter
    import pandas as pd

    import wandb.apis.reports as wr

    api = wandb.Api()


def get_task_description(task):
    clazz = TASK_REGISTRY[task]
    task_doc = inspect.cleandoc(inspect.getdoc(inspect.getmodule(clazz)))
    task_citation = getattr(inspect.getmodule(clazz), "_CITATION")
    return task_doc, task_citation


def log_results(run, results):
    name = run.name
    result_dict = results.copy()
    wandb.config.update(result_dict["config"])
    fewshot = result_dict["config"]["num_fewshot"]
    task_results = defaultdict(lambda: defaultdict(list))
    columns = ["Task", "Version", "num_fewshot", "Metric", "Value", "Stderr"]
    values = []
    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue
            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, fewshot, m, v, se])
                task_results[k][m].extend(["%.4f" % v, "%.4f" % se])
            else:
                values.append([k, version, fewshot, m, v, None])
                task_results[k][m].extend(["%.4f" % v, ""])
    results_table = wandb.Table(columns=columns, data=values)
    run_artifact = wandb.Artifact(name, type="results")
    run_artifact.add(results_table, "results_table")
    with run_artifact.new_file("results_json", "w+") as f:
        json.dump(result_dict, f)
    run.log({f"{name}_results_table": results_table})
    run.log_artifact(run_artifact)
    return task_results


def write_metric_table(metrics):
    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Metric", "Value", "± Stderr"]
    md_values = []
    for k, v in metrics.items():
        md_values.append([f"**{k}**"] + v)
    md_writer.value_matrix = md_values
    return md_writer.dumps()


def create_report_by_task(task_results):
    tasks_report = []
    task_descriptions = {}
    for task, metrics in task_results.items():
        tasks_report.append(wr.H2(task.title()))
        task_description, task_citation = get_task_description(task)
        if task_description:
            task_descriptions[task] = task_description
        tasks_report.extend(
            [wr.H3("Metrics"), wr.MarkdownBlock(write_metric_table(metrics))]
        )
    return tasks_report, task_descriptions


def get_model_name_from_args(args):
    model_args = simple_parse_args_string(args.model_args)

    model_name = args.model
    if args.model == "gpt2":
        model_type = model_args.get("pretrained", "")
        if model_type:
            model_name = model_type
    elif args.model == "gpt3":
        model_type = model_args.get("engine", "")
        if model_type:
            model_name = model_name + ":" + model_type
    return model_name


def report_evaluation(
    args, results, results_md,
):
    if not WANDB_AVAILABLE:
        return
    if args.wandb_project is None:
        return
    else:
        wandb_project = args.wandb_project
    if wandb.run is None:
        wandb_entity = (
            args.wandb_entity
            if args.wandb_entity is not None
            else wandb.apis.PublicApi().default_entity
        )
        wandb_group = args.wandb_group
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type="evaluation",
            group=wandb_group,
        )
    else:
        run = wandb.run
        wandb_project = run.project
        wandb_entity = run.entity

    task_results = log_results(run, results,)
    tasks_report, task_descriptions = create_report_by_task(task_results)
    model_name = get_model_name_from_args(args)
    report = wr.Report(
        project=wandb_project,
        entity=wandb_entity,
        title=f"({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}) Model: {model_name} - Evaluation report",
        description=f"Evaluation run by: {run.entity} logged to {run.url}",
    )

    blocks = (
        [
            wr.TableOfContents(),
            wr.H1("Complete Evaluation Results"),
            wr.MarkdownBlock(results_md),
            wr.H1("Evaluation Results By Task"),
        ]
        + tasks_report
        + [
            wr.H1("Evaluation Runs"),
            wr.WeaveBlockSummaryTable(
                project=wandb_project,
                entity=wandb_entity,
                table_name=f"{run.name}_results_table",
            ),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(
                        project=wandb_project, entity=wandb_entity,
                    ).set_filters_with_python_expr(f'Name == "{str(run.name)}"'),
                ]
            ),
            wr.H1("Evaluation Config"),
            wr.CodeBlock(
                json.dumps(results["config"], indent=5).split("\n"), language="json"
            ),
        ]
    )
    blocks.append(wr.H1("Task Appendix"))
    for task in task_descriptions:
        blocks.append(wr.H2(task.title()))
        blocks.append(wr.MarkdownBlock(task_descriptions[task]))
    report.blocks = blocks
    report.save()
    print(f"Check out the autogenerated report at: {report.url}")
    wandb.finish()


def fetch_runs(project, group, job_type):
    runs = api.runs(
        project,
        {"$and": [{"group": {"$eq": group}}, {"jobType": {"$eq": job_type}}]},
        include_sweeps=False,
    )

    runs_df = []
    run_configs = []
    for run in runs:
        artifact_name = f"{run.entity}/{run.project}/{run.name}:latest"
        artifact_table = "results_table.table.json"
        run_table = api.artifact(artifact_name).get(artifact_table)
        model = simple_parse_args_string(run.config["model_args"]).get(
            "pretrained", run.config["model"]
        )
        run_df = pd.DataFrame(run_table.data, columns=run_table.columns)
        # this currently only works for gpt2 models, Need to extend it to other models
        num_parameters = (
            get_model(run.config["model"])
            .create_from_arg_string(run.config["model_args"], {"device": "cpu"})
            .gpt2.num_parameters()
        )
        run_df["Model"] = model
        run_df["Parameters"] = num_parameters
        runs_df.append(run_df)
        run_configs.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
    runs_df = pd.concat(runs_df)

    return runs_df, run_configs


def log_model_comparisons(run,):
    runs_df, run_configs = fetch_runs(run.project, run.group, job_type="evaluation")
    task_titles = []
    for name, group in runs_df.groupby(["Task", "Version", "Metric"]):
        task_title = f"{name[0]}:{name[1]}:{name[2]}"
        task_data = group[["Model", "Parameters", "Value", "Stderr"]]
        task_data = task_data.rename({"Value": name[2]}, axis=1)
        task_data = task_data.sort_values(["Parameters"])
        task_table = wandb.Table(dataframe=task_data)
        fields = {
            "x": task_table.columns[1],
            "y": task_table.columns[2],
            "model": task_table.columns[0],
            "title": task_title,
        }
        task_plot = wandb.plot_table(
            vega_spec_name=COMPARE_MODELS_VEGA_SPEC,
            data_table=task_table,
            fields=fields,
        )
        # run.log({f"{task_title}_table": task_table})
        run.log({f"{task_title}_plot": task_plot})
        task_titles.append(task_title)
    return task_titles, run_configs


def report_compared_models(args):
    if not WANDB_AVAILABLE:
        return
    if args.wandb_project is None:
        return
    else:
        wandb_project = args.wandb_project
    if wandb.run is None:
        wandb_entity = (
            args.wandb_entity
            if args.wandb_entity is not None
            else wandb.apis.PublicApi().default_entity
        )
        wandb_group = args.wandb_group
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type="compare_models",
            group=wandb_group,
        )
    else:
        run = wandb.run
        wandb_project = run.project
        wandb_entity = run.entity

    task_titles, run_configs = log_model_comparisons(run,)
    report = wr.Report(
        project=wandb_project,
        entity=wandb_entity,
        title=f"({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}) Model comparison report",
        description=f"Evaluation run by: {run.entity} logged to {run.url}",
    )

    panel_grid = wr.PanelGrid(
        runsets=[
            wr.Runset(project=wandb_project, entity=wandb_entity, query=run.name,)
        ],
        panels=[
            wr.WeavePanelSummaryTable(table_name=f"{task_title}_plot_table")
            for task_title in task_titles
        ],
    )
    blocks = [
        wr.H1("Model Comparison Report"),
        wr.CalloutBlock(
            [
                f"The plots for this report are generated in the wandb run: {wr.Link(text=run.name, url=run.url)}",
                "To import the plots into this report, do the following:",
                "\t - Go to the chart you want to import in the wandb run above",
                "\t - Click on vertical ellipsis (...) on the top right corner of the chart",
                "\t - Select the 'Add to report' option and add the chart to this report",
            ]
        ),
        panel_grid,
        wr.H1("Evaluation Configs"),
    ]
    for config in run_configs:
        model = simple_parse_args_string(config["model_args"]).get(
            "pretrained", config["model"]
        )
        blocks.append(wr.H2(f"{model}"))
        blocks.append(
            wr.CodeBlock(json.dumps(config, indent=5).split("\n"), language="json")
        )
    wandb.finish()
    report.blocks = blocks
    report.save()
    print(f"Check out the autogenerated report at: {report.url}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", default=None, type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_group", default=None, type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    report_compared_models(args)


if __name__ == "__main__":
    main()
