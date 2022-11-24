import logging

logger = logging.getLogger(__name__)
WANDB_AVAILABLE = False
try:
    import wandb

    # wandb.require("report-editing:v0")
    WANDB_AVAILABLE = True
except:
    logger.warning(
        "To use the wandb reporting functionality please install wandb>=0.13.6.\n"
        "install the latest version of wandb with `pip install wandb --upgrade`"
    )
    WANDB_AVAILABLE = False

if WANDB_AVAILABLE:
    from collections import defaultdict

    from pytablewriter import MarkdownTableWriter

    import wandb.apis.reports as wr
    from lm_eval import utils, tasks
    import json
    from datetime import datetime
    import inspect


def get_task_description(task):
    clazz = tasks.TASK_REGISTRY[task]
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
    run.log({name: results_table})
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
    model_args = utils.simple_parse_args_string(args.model_args)

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
        run = wandb.init(project=wandb_project, entity=wandb_entity)
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
        title=f"Evaluation report for {model_name} model run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        description=f"Evaluation run by: {run.entity} results logged to {run.url}",
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
            # TODO: Add WeaveTableBlock once it becomes available in wandb
            # wr.WeaveTableBlock(
            #     project=wandb_project, entity=wandb_entity, table_name=f"{run.name}",
            # ),
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
    for task, description in task_descriptions.items():
        blocks.extend([wr.H2(task.title()), wr.P(description)])
    report.blocks = blocks
    report.save()
    print(f"Check out the autogenerated report at: {report.url}")
