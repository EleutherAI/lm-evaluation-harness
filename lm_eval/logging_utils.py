import copy
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

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


def remove_none_pattern(input_string: str) -> Tuple[str, bool]:
    """Remove the ',none' substring from the input_string if it exists at the end.

    Args:
        input_string (str): The input string from which to remove the ',none' substring.

    Returns:
        Tuple[str, bool]: A tuple containing the modified input_string with the ',none' substring removed
                          and a boolean indicating whether the modification was made (True) or not (False).
    """
    # Define the pattern to match ',none' at the end of the string
    pattern = re.compile(r",none$")

    # Use sub() to replace ',none' with an empty string
    result = re.sub(pattern, "", input_string)

    # check if the input_string changed
    removed = result != input_string

    return result, removed


def _handle_non_serializable(o: Any) -> Union[int, str, list]:
    """Handle non-serializable objects by converting them to serializable types.

    Args:
        o (Any): The object to be handled.

    Returns:
        Union[int, str, list]: The converted object. If the object is of type np.int64 or np.int32,
            it will be converted to int. If the object is of type set, it will be converted
            to a list. Otherwise, it will be converted to str.
    """
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


class WandbLogger:
    def __init__(self, results: Dict[str, Any], args: Any) -> None:
        """Initialize the WandbLogger.

        Args:
            results (Dict[str, Any]): The results dictionary.
            args (Any): Arguments for configuration.
        """
        self.results: Dict[str, Any] = copy.deepcopy(results)
        self.wandb_args: Dict[str, Any] = utils.simple_parse_args_string(
            args.wandb_args
        )

        self.task_names: List[str] = list(results.get("results", {}).keys())

        # initialize a W&B run
        if wandb.run is None:
            self.run = wandb.init(**self.wandb_args)
        else:
            self.run = wandb.run

    def log_eval_result(self) -> None:
        """Log evaluation results to W&B."""
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

    def get_eval_wandb_table(self) -> None:
        """Generate and log evaluation results as a table to W&B."""
        columns: List[str] = [
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

    def generate_dataset(
        self, data: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate a dataset from evaluation data.

        Args:
            data (List[Dict[str, Any]]): The data to generate a dataset for.
            config (Dict[str, Any]): The configuration of the task.

        Returns:
            pd.DataFrame: A dataframe that is ready to be uploaded to W&B.
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

    def log_eval_samples(self, samples: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log evaluation samples to W&B.

        Args:
            samples (Dict[str, List[Dict[str, Any]]]): Evaluation samples for each task.
        """
        for task_name in self.task_names:
            eval_preds = samples[task_name]

            # log the samples as an artifact
            dumped = json.dumps(
                eval_preds,
                indent=2,
                default=_handle_non_serializable,
                ensure_ascii=False,
            )
            artifact = wandb.Artifact(f"{task_name}", type="samples_by_task")
            with artifact.new_file(
                f"{task_name}_eval_samples.json", mode="w", encoding="utf-8"
            ) as f:
                f.write(dumped)
            self.run.log_artifact(artifact)
            artifact.wait()

            # log the samples as a W&B Table
            df = self.generate_dataset(eval_preds, self.task_configs.get(task_name))
            self.run.log({f"{task_name}_eval_results": df})

    def log_results_as_json(self) -> None:
        """Log results as JSON artifact to W&B."""
        dumped = json.dumps(
            self.results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        artifact = wandb.Artifact("results", type="eval_results")
        with artifact.new_file("results.json", mode="w", encoding="utf-8") as f:
            f.write(dumped)
        self.run.log_artifact(artifact)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        self.task_configs = self.results.get("configs", {})
        cli_configs = self.results.get("config", {})
        configs = {
            "task_configs": self.task_configs,
            "cli_configs": cli_configs,
        }

        return configs

    def sanitize_results_dict(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Sanitize the results dictionary."""
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

    def prepare_report_by_task(self, results: Dict[str, Any]) -> List[Any]:
        """Prepare report by task."""
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

    def _write_to_report(self) -> None:
        """Write to report."""
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

    def write_to_report(self) -> None:
        try:
            self._write_to_report()
        except Exception as e:
            wandb.termerror(
                f"The program failed to automatically generate a W&B Report due to {e} ."
                "Please head over to https://docs.wandb.ai/guides/reports/create-a-report to learn "
                "how to create a report in the UI."
            )
