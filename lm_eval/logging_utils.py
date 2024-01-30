import copy
import logging
import re
import json
from datetime import datetime

from packaging.version import Version

from lm_eval import utils

logger = logging.getLogger(__name__)

IS_WANDB_AVAILABLE = False


try:
    import wandb

    assert Version(wandb.__version__) >= Version("0.13.6")
    if Version(wandb.__version__) < Version("0.13.6"):
        wandb.require("report-editing:v0")
    IS_WANDB_AVAILABLE = True
except Exception as e:
    logger.warning(
        "To use the wandb reporting functionality please install wandb>=0.13.6.\n"
        "To install the latest version of wandb run `pip install wandb --upgrade`\n"
        f"{e}"
    )
    IS_WANDB_AVAILABLE = False

if IS_WANDB_AVAILABLE:
    import wandb.apis.reports as wr


def remove_none_pattern(input_string):
    # Define the pattern to match ',none' at the end of the string
    pattern = re.compile(r",none$")

    # Use sub() to replace ',none' with an empty string
    result = re.sub(pattern, "", input_string)

    # check if the input_string changed
    removed = result != input_string

    return result, removed


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dictionary.

    Parameters:
    - d (dict): The nested dictionary to be flattened.
    - parent_key (str, optional): The key from the parent dictionary (used for recursion). Defaults to an empty string.
    - sep (str, optional): The separator used between keys when generating the flattened keys. Defaults to '_'.

    Returns:
    - dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def remove_keys_with_substrings(d, substrings_to_remove):
    """
    Remove keys containing specified substrings from a dictionary.

    Parameters:
    - d (dict): The original dictionary.
    - substrings_to_remove (list): List of substrings to be removed from keys.

    Returns:
    - dict: The modified dictionary.
    """
    return {
        key: value
        for key, value in d.items()
        if not any(substring in key for substring in substrings_to_remove)
    }


class WandbLogger:
    def __init__(self, results, args):
        self.results = copy.deepcopy(results)
        self.wandb_args = utils.simple_parse_args_string(args.wandb_args)

        self.task_names = list(results.get("results", {}).keys())

    def log_eval_result(self):
        # initialize a W&B run
        self.run = wandb.init(**self.wandb_args)

        # Log configs to wandb
        configs = self.get_config()
        self.run.config.update(configs)

        wandb_summary, self.wandb_results = self.sanitize_results_dict()
        # update wandb.run.summary with items that were removed
        self.run.summary.update(wandb_summary)
        # Log the evaluation metrics to wandb
        self.run.log(self.wandb_results)
        # Log the evaluation metrics as Table
        self.get_eval_wandb_table()

    def get_eval_wandb_table(self):
        columns = ["Task", "Version", "Filter", "num_fewshot", "Metric", "Value", "Stderr"]
        table = wandb.Table(columns=columns)
        results = copy.deepcopy(self.results)

        for k, dic in results.get("results").items():
            version = results.get("versions").get(k)
            n = results.get("n-shot").get(k)

            if "alias" in dic:
                k = dic.pop("alias")

            for (mf), v in dic.items():
                m, _, f = mf.partition(",")
                if m.endswith("_stderr"):
                    continue

                if m + "_stderr" + "," + f in dic:
                    se = dic[m + "_stderr" + "," + f]
                    if se != "N/A":
                        se = "%.4f" % se
                    table.add_data(*[k, version, f, n, m, v, se])
                else:
                    table.add_data(*[k, version, f, n, m, v, ""])

        # log the table to W&B
        self.run.log({f"evaluation/eval_results": table})

    def log_eval_samples(self, samples):
        assert self.run is not None

        for task_name in self.task_names:
            eval_preds = samples[task_name]

            _eval_preds = []
            for eval_pred in eval_preds:
                eval_pred = flatten_dict(eval_pred)
                eval_pred = remove_keys_with_substrings(
                    eval_pred,
                    substrings_to_remove=[
                        "resps",
                    ],
                )
                _eval_preds.append(eval_pred)

            # initialize a new W&B Table
            columns = list(_eval_preds[0].keys())
            table = wandb.Table(columns=columns)
            # TODO: handle columns with list data in a better way
            for _eval_pred in _eval_preds:
                table.add_data(*_eval_pred.values())

            # log the table to W&B
            self.run.log({f"{task_name}_eval_results": table})

            del _eval_preds

    def get_config(self):
        task_configs = self.results.get("configs", {})
        cli_configs = self.results.get("config", {})
        configs = {
            "task_configs": task_configs,
            "cli_configs": cli_configs,
        }

        return configs

    def sanitize_results_dict(self):
        """
        Remove string valued keys from the results dict as they don't render in the workspace.
        Log these key-value pairs to wandb.summary.
        """
        _results = copy.deepcopy(self.results.get("results", dict()))

        # Remove None from the metric string name
        tmp_results = copy.deepcopy(_results)
        for task_name in self.task_names:
            task_result = tmp_results.get(task_name, dict())
            for metric_name, metric_value in task_result.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if removed:
                    _results[task_name][_metric_name] = metric_value
                    _results[task_name].pop(metric_name)

        # remove string valued keys from the results dict
        wandb_summary = {}
        for task in self.task_names:
            task_result = _results.get(task, dict())
            for metric_name, metric_value in task_result.items():
                if isinstance(metric_value, str):
                    wandb_summary[f"{task}/{metric_name}"] = metric_value

        for summary_metric, summary_value in wandb_summary.items():
            _task, _summary_metric = summary_metric.split("/")
            _results[_task].pop(_summary_metric)

        tmp_results = copy.deepcopy(_results)
        for task_name, task_results in tmp_results.items():
            for metric_name, metric_value in task_results.items():
                _results[f"{task_name}/{metric_name}"] = metric_value
                _results[task_name].pop(metric_name)
        for task in self.task_names:
            _results.pop(task)

        return wandb_summary, _results

    def prepare_report_by_task(self, results):
        _results = results.get("results", dict())

        blocks = []
        for task_name in self.task_names:
            blocks.append(wr.H2(task_name))
            panels = []
            for metric_name, metric_value in results.items():
                if task_name in metric_name:
                    panels.append(
                        wr.ScalarChart(
                            title=f"{metric_name}",
                            metric=f"{metric_name}",
                            font_size="large",
                        )
                    )
            blocks.append(
                wr.PanelGrid(panels=panels)
            )

        return blocks

    def write_to_report(self):
        wandb_project = self.wandb_args.get("project", "lm-eval-harness")
        wandb_entity = self.wandb_args.get("entity", None)
        report = wr.Report(
            project=wandb_project,
            entity=wandb_entity,
            title=f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) xxx - Evaluation report",
            description=f"Evaluation run by: {self.run.entity} logged to {self.run.url}",
        )

        results_md = utils.make_table(self.results)
        task_blocks = self.prepare_report_by_task(self.wandb_results)

        blocks = (
            [
                wr.TableOfContents(),
                wr.H1("Complete Evaluation Results"),
                wr.MarkdownBlock(results_md),
                wr.H1("Evaluation Results By Task"),
            ]
            + task_blocks
            + [
                wr.H1("Evaluation Runs"),
                # wr.WeaveBlockSummaryTable(
                #     project=wandb_project,
                #     entity=wandb_entity,
                #     table_name=f"{run.name}_results_table",
                # ),
                # wr.PanelGrid(
                #     runsets=[
                #         wr.Runset(
                #             project=wandb_project, entity=wandb_entity,
                #         ).set_filters_with_python_expr(f'Name == "{str(run.name)}"'),
                #     ]
                # ),
                wr.PanelGrid(
                    runsets=[
                        wr.Runset(
                            project=wandb_project, entity=wandb_entity,
                        ).set_filters_with_python_expr(f'Name == "{str(self.run.name)}"'),
                    ]
                ),
                wr.H1("Evaluation Config"),
                wr.CodeBlock(
                    json.dumps(self.results["config"], indent=5).split("\n"), language="json"
                ),
                wr.H1("Task Appendix")
            ]
        )

        report.blocks = blocks
        report.save()
        wandb.termlog(f"📝 Check out the autogenerated report at: {report.url}")
