import os
import glob
import json
import pandas as pd
import wandb


def collect_jsons(path, custom_cols=None):
    """
    Collect all json files into a single dataframe.

    Args:
        path (str): Path to the root folder containing the json files.
        custom_cols (list of strings or None): If specified, only specified custom columns are used. Else, all are used.

    Returns:
        lmplot object.
    """

    json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)

    data_records = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        results = data["results"]
        versions = data["versions"]
        config = data["config"]
        model = config["model"]

        try:
            custom_info = data["custom_info"]

            if custom_cols:
                custom_info = {
                    key: value
                    for key, value in custom_info.items()
                    if key in custom_cols
                }

        except:
            print(f"Warning: Could not find custom info for {json_file}.")
            custom_info = {}

        for task in results:
            for metric in results[task]:
                record = {
                    "model": str(model),
                    "task": str(task),
                    "task_version": str(versions[task]),
                    "metric": str(metric),
                    "value": float(results[task][metric]),
                }
                record.update(custom_info)
                data_records.append(record)

    df = pd.DataFrame(data_records)
    data_records.clear()
    df.reset_index(inplace=True, drop=True)

    if df.isnull().values.any():
        nan_columns = df.columns[df.isnull().any()].tolist()
        print(
            f"Warning: Found nan values. Check if all json files contain the {nan_columns}."
        )

    return lmplot(df)


class lmplot:
    def __init__(self, df):
        self.df = df

    def get_df(self):
        return self.df

    def all_models(self):
        """
        Return a list of all models in the data frame.
        """

        return self.df["model"].unique().tolist()

    def filter_df(self, x="step", model=None, task=None, metric=None, to_csv=None):
        """
        Filter the data frame for the specified model, task, metric.

        Args:
            model (string or list of strings or None): If specified, only return info for the specified model(s).
            task (string or list of strings or None): If specified, only return info for the specified task(s).
            metric (string or list of strings or None): If specified, only return info for the specified metric(s).

        Returns:
            Dataframe with columns model, task, metric.
        """

        df = self.df

        if x not in df.columns:
            raise ValueError(f"{x} not found in columns {df.columns.tolist()}")

        try:
            df[x] = df[x].astype(int)
        except:
            print(f"Warning: {x} is not an integer column.")

        basic_cols = ["model", "task", "task_version", "metric", x]

        if df.duplicated(basic_cols).all():
            raise ValueError(
                "Found duplicate row for columns: {basic_cols}. Ensure they are unique for each json file (Specially model name)."
            )

        filters = {}
        filters["model"] = model if isinstance(model, list) else [model]
        filters["task"] = task if isinstance(task, list) else [task]
        filters["metric"] = metric if isinstance(metric, list) else [metric]

        for colname, colvalues in filters.items():

            if task is None and colname == "task":
                # If task is not specified, all tasks are plotted
                colvalues = df["task"].unique().tolist()

            if metric is None and colname == "metric":
                # If metric is not specified, all metrics are plotted
                colvalues = df["metric"].unique().tolist()

            invalid_colvalues = [
                c for c in colvalues if c not in df[colname].unique().tolist()
            ]

            if invalid_colvalues:

                if colname == "task":
                    # Filter out tasks that are substrings of other tasks. Example: task="math" will get all tasks containing "math".

                    for invalid_task in invalid_colvalues:
                        invalid_flag = True

                        for task in df["task"].unique().tolist():
                            if invalid_task in task:
                                colvalues.append(task)
                                invalid_flag = False

                        if invalid_flag:
                            raise ValueError(
                                f"{colname} names {invalid_task} not found. Available are {df[colname].unique().tolist()}"
                            )

                else:
                    raise ValueError(
                        f"{colname} names {invalid_colvalues} not found. Available are {df[colname].unique().tolist()}"
                    )

            df = df[df[colname].isin(colvalues)]

        df = df.reset_index(drop=True)
        df = df.sort_values(by=x)

        if to_csv:
            df.to_csv(to_csv, index=False)

        return df

    def _lineplot_tasks(
        self,
        df,
        x="step",
        model=None,
        task=None,
        metric=None,
        hue="model",
        compare=False,
    ):

        """
        Filter the dataframe and plot the lineplot for each.
        """

        for task in df["task"].unique().tolist():
            task = str(task)
            task_df = df[df["task"] == task]
            task_metrics = task_df["metric"].unique().tolist()

            for metric in task_metrics:
                metric = str(metric)

                metric_df = task_df[task_df["metric"] == metric]

                metric_df = metric_df.sort_values(by=x)

                table = wandb.Table(dataframe=metric_df)
                fields = {
                    "x-axis": f"{x}",
                    "y-axis": "value",
                    "color": f"{hue}",
                    "metric": f"{metric}",
                    "title": f"{task} ({metric})",
                }

                custom_chart = wandb.plot_table(
                    vega_spec_name="satpalsr/multiplot", data_table=table, fields=fields
                )

                if compare:

                    if len(task_metrics) == 1:
                        wandb.log({f"{task} {metric}": custom_chart})
                    else:
                        wandb.log({f"{task}/{metric}": custom_chart})

                else:
                    wandb.log({f"{model}/{task} ({metric})": custom_chart})

    def lineplot(
        self,
        x="step",
        model=None,
        task=None,
        metric=None,
        hue="model",
        compare=False,
        **kwargs,
    ):

        """
        Draw lineplot for each model, task and metric combination.

        Args:
            x (str): x-axis column name.
            model (str or list of strings or None): List of model names.
            task (str or list of strings or None): List of task names.
            metric (str or list of strings or None): List of metric names.
            hue (str): Column name for hue
            compare (bool): If True, Models are compared in each plot. Plots are saved in task folders.
                            Else, Models are not compared. Plots are saved in model folders.

        Returns:
            None
        """

        if model is None:
            model = self.all_models()

        df = self.filter_df(x, model, task, metric)

        if df.empty:
            raise ValueError("model, task, metric combination not found.")

        run = wandb.init(**kwargs)

        if compare:
            self._lineplot_tasks(df, x, model, task, metric, hue, compare)

            run.finish()

        else:

            for model in df["model"].unique().tolist():
                model_df = df[df["model"] == model]

                self._lineplot_tasks(model_df, x, model, task, metric, hue, compare)

            run.finish()
