# TODO: Silent Wandb and just display run and reports url
import copy
import json
import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
from packaging.version import Version

from lm_eval import utils


logger = logging.getLogger(__name__)

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


def remove_none_pattern(input_string):
    # Define the pattern to match ',none' at the end of the string
    pattern = re.compile(r",none$")

    # Use sub() to replace ',none' with an empty string
    result = re.sub(pattern, "", input_string)

    # check if the input_string changed
    removed = result != input_string

    return result, removed


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


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
        # Log the results dict as json
        self.log_results_as_json()

    def get_eval_wandb_table(self):
        columns = [
            "Task",
            "Version",
            "Filter",
            "num_fewshot",
            "Metric",
            "Value",
            "Stderr",
        ]
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
        self.run.log({"evaluation/eval_results": table})

    def generate_dataset(self, data, config):
        """Generate a Zeno dataset from evaluation data.

        Args:
            data: The data to generate a dataset for.
            config: The configuration of the task.

        Returns:
            pd.Dataframe: A dataframe that is ready to be uploaded to Zeno.
        """
        ids = [x["doc_id"] for x in data]
        labels = [x["target"] for x in data]
        instance = [""] * len(ids)

        metrics_list = config["metric_list"]
        metrics = {}
        for metric in metrics_list:
            metric = metric.get("metric")
            metrics[metric] = [x[metric] for x in data]

        if config["output_type"] == "loglikelihood":
            instance = [x["arguments"][0][0] for x in data]
            labels = [x["arguments"][0][1] for x in data]
        elif config["output_type"] == "multiple_choice":
            instance = [
                x["arguments"][0][0]
                + "\n\n"
                + "\n".join([f"- {y[1]}" for y in x["arguments"]])
                for x in data
            ]
        elif config["output_type"] == "loglikelihood_rolling":
            instance = [x["arguments"][0][0] for x in data]
        elif config["output_type"] == "generate_until":
            instance = [x["arguments"][0][0] for x in data]

        df_data = {
            "id": ids,
            "data": instance,
            "input_len": [len(x) for x in instance],
            "labels": labels,
            "output_type": config["output_type"],
        }
        df_data.update(metrics)

        return pd.DataFrame(df_data)

    def log_eval_samples(self, samples):
        for task_name in self.task_names:
            eval_preds = samples[task_name]
            df = self.generate_dataset(eval_preds, self.task_configs.get(task_name))
            self.run.log({f"{task_name}_eval_results": df})

    def log_results_as_json(self):
        dumped = json.dumps(
            self.results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        artifact = wandb.Artifact("results", type="eval_results")
        with artifact.new_file("results.json", mode="w", encoding="utf-8") as f:
            f.write(dumped)
        self.run.log_artifact(artifact)

    def get_config(self):
        self.task_configs = self.results.get("configs", {})
        cli_configs = self.results.get("config", {})
        configs = {
            "task_configs": self.task_configs,
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
        import wandb.apis.reports as wr

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
            _results = {
                "results": {f"{task_name}": self.results.get("results").get(task_name)},
                "versions": {
                    f"{task_name}": self.results.get("versions").get(task_name)
                },
                "n-shot": {f"{task_name}": self.results.get("n-shot").get(task_name)},
            }
            results_md = utils.make_table(_results)
            blocks.extend([wr.MarkdownBlock(results_md), wr.PanelGrid(panels=panels)])
            # TODO: Add results table

        return blocks

    def write_to_report(self):
        import wandb.apis.reports as wr

        report = wr.Report(
            project=self.run.project,
            entity=self.run.entity,
            title=f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) {self.run.id} - Evaluation report",
            description=f"Evaluation run by: {self.run.entity} logged to {self.run.url}",
        )

        task_blocks = self.prepare_report_by_task(self.wandb_results)

        blocks = (
            [
                wr.TableOfContents(),
                wr.H1("Complete Evaluation Results"),
                wr.WeaveBlockSummaryTable(
                    project=self.run.project,
                    entity=self.run.entity,
                    table_name="evaluation/eval_results",
                ),
                wr.PanelGrid(
                    runsets=[
                        wr.Runset(
                            project=self.run.project,
                            entity=self.run.entity,
                        ).set_filters_with_python_expr(
                            f'Name == "{str(self.run.name)}"'
                        ),
                    ]
                ),
                wr.H1("Evaluation Results By Task"),
            ]
            + task_blocks
            + [
                wr.H1("Evaluation Config"),
                wr.CodeBlock(
                    json.dumps(self.results["config"], indent=5).split("\n"),
                    language="json",
                ),
                # TODO: Add appendix
            ]
        )

        report.blocks = blocks
        report.save()
        wandb.termlog(f"📝 Check out the autogenerated report at: {report.url}")
