import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from huggingface_hub import (
    HfApi,
)

from lm_eval.tracking.info_trackers import (
    #     DetailsTracker,
    GeneralConfigTracker,
    #     MetricsTracker,
    #     TaskConfigTracker,
    VersionsTracker,
)
from lm_eval.utils import eval_logger


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Provides a proper json encoding for the loggers and trackers json dumps.
    Notably manages the json encoding of dataclasses.
    """

    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


class EvaluationTracker:
    """Keeps track of the overall evaluation process and relevant information.

    The [`EvaluationTracker`] contains specific loggers for experiments details ([`DetailsLogger`]), metrics ([`MetricsLogger`]), task versions ([`VersionsLogger`]) as well as for the general configurations of both the specific task ([`TaskConfigLogger`]) and overall evaluation run ([`GeneralConfigLogger`]).
    It compiles the data from these loggers and writes it to files, which can be published to the Hugging Face hub if requested.
    """

    # details_tracker: DetailsTracker
    # metrics_tracker: MetricsTracker
    versions_tracker: VersionsTracker
    general_config_tracker: GeneralConfigTracker
    # task_config_tracker: TaskConfigTracker
    hub_results_org: str

    def __init__(self, hub_results_org: str = "", token: str = "") -> None:
        """Creates all the necessary loggers for evaluation tracking.

        Args:
            hub_results_org (str): The organisation to push the results to. See more details about the datasets organisation in [`EvaluationTracker.save`]
            token (str): Token to use when pushing to the hub. This token should have write access to `hub_results_org`.
        """
        # self.details_tracker = DetailsTracker()
        # self.metrics_tracker = MetricsTracker()
        self.versions_tracker = VersionsTracker()
        self.general_config_tracker = GeneralConfigTracker()
        # self.task_config_tracker = TaskConfigTracker()
        self.hub_results_org = hub_results_org
        self.hub_results_repo = f"{hub_results_org}/results"
        self.hub_private_results_repo = f"{hub_results_org}/results-private"
        self.api = HfApi(token=token)

    def save(
        self,
        output_dir: str,
        push_results_to_hub: bool,
        push_details_to_hub: bool,
        public: bool,
        # push_results_to_tensorboard: bool = False,
    ) -> None:
        """Saves the experiment information and results to files, and to the hub if requested.

        Note: In case of save failure, this function will only print a warning, with the error message.

        Args:
            output_dir (str): Local folder path where you want results to be saved
            push_results_to_hub (bool): If True, results are pushed to the hub.
                Results will be pushed either to `{hub_results_org}/results`, a public dataset, if `public` is True else to `{hub_results_org}/private-results`, a private dataset.
            push_details_to_hub (bool): If True, details are pushed to the hub.
                Results are pushed to `{hub_results_org}/details__{sanitized model_name}` for the model `model_name`, a public dataset,
                if `public` is True else `{hub_results_org}/details__{sanitized model_name}_private`, a private dataset.
            public (bool): If True, results and details are pushed in private orgs

        """
        eval_logger.info("Saving experiment tracker")
        try:
            date_id = datetime.now().isoformat().replace(":", "-")

            model_name_sanitized = self.general_config_tracker.model_name.replace(
                "/", "__"
            )
            output_dir_results = Path(output_dir) / "results" / model_name_sanitized
            output_dir_details = Path(output_dir) / "details" / model_name_sanitized
            output_dir_details_sub_folder = output_dir_details / date_id
            output_dir_results.mkdir(parents=True, exist_ok=True)
            output_dir_details_sub_folder.mkdir(parents=True, exist_ok=True)

            output_results_file = output_dir_results / f"results_{date_id}.json"
            output_results_in_details_file = (
                output_dir_details / f"results_{date_id}.json"
            )

            eval_logger.info(
                f"Saving results to {output_results_file} and {output_results_in_details_file}"
            )

            to_dump = {
                "config_general": asdict(self.general_config_tracker),
                #     # "results": self.metrics_tracker.metric_aggregated,
                "versions": self.versions_tracker.versions,
                #     # "config_tasks": self.task_config_tracker.tasks_configs,
                #     # "summary_tasks": self.details_tracker.compiled_details,
                #     # "summary_general": asdict(self.details_tracker.compiled_details_over_all_tasks),
            }
            dumped = json.dumps(to_dump, cls=EnhancedJSONEncoder, indent=2)

            eval_logger.info(f"output results file - tracker: {output_results_file}")
            with open(output_results_file, "w") as f:
                f.write(dumped)

            with open(output_results_in_details_file, "w") as f:
                f.write(dumped)

            # for task_name, task_details in self.details_tracker.details.items():
            #     output_file_details = output_dir_details_sub_folder / f"details_{task_name}_{date_id}.parquet"
            #     # Create a dataset from the dictionary
            #     try:
            #         dataset = Dataset.from_list([asdict(detail) for detail in task_details])
            #     except Exception:
            #         # We force cast to str to avoid formatting problems for nested objects
            #         dataset = Dataset.from_list(
            #             [{k: str(v) for k, v in asdict(detail).items()} for detail in task_details]
            #         )

            #     # We don't keep 'id' around if it's there
            #     column_names = dataset.column_names
            #     if "id" in dataset.column_names:
            #         column_names = [t for t in dataset.column_names if t != "id"]

            #     # Sort column names to make it easier later
            #     dataset = dataset.select_columns(sorted(column_names))
            #     # Save the dataset to a Parquet file
            #     dataset.to_parquet(output_file_details.as_posix())

            if push_results_to_hub:
                self.api.create_repo(
                    repo_id=self.hub_results_repo,
                    repo_type="dataset",
                    private=not self.public,
                    exist_ok=True,
                )
                self.api.upload_folder(
                    repo_id=self.hub_results_repo
                    if public
                    else self.hub_private_results_repo,
                    folder_path=output_dir_results,
                    path_in_repo=self.general_config_tracker.model_name,
                    repo_type="dataset",
                    commit_message=f"Updating model {self.general_config_tracker.model_name}",
                )

            # if push_details_to_hub:
            #     self.details_to_hub(
            #         model_name=self.general_config_tracker.model_name,
            #         results_file_path=output_results_in_details_file,
            #         details_folder_path=output_dir_details_sub_folder,
            #         push_as_public=public,
            #     )

            # if push_results_to_tensorboard:
            #     self.push_results_to_tensorboard(
            #         results=self.metrics_tracker.metric_aggregated, details=self.details_tracker.details
            #     )
        except Exception as e:
            eval_logger.info("WARNING: Could not save results")
            eval_logger.info(repr(e))
