import os
import time
from dataclasses import dataclass
from typing import Optional, Union

import git


# from lighteval.metrics import MetricCategory
# from lighteval.metrics.stderr import get_stderr_function
# from lighteval.models.model_loader import ModelInfo

# from lighteval.models.model_output import ModelReturn
# from lighteval.tasks.lighteval_task import LightevalTask
# from lighteval.tasks.requests import Doc


@dataclass(init=False)
class GeneralConfigTracker:
    """Tracker for the evaluation parameters.

    Attributes:
        lighteval_sha (str): Current commit sha of lighteval used for the evaluation (for reproducibility purposes)
        override_batch_size (int): Manages the batch size.
            If strictly positive, its value is used as the batch size for all experiments.
            Else, the batch size is automatically inferred depending on what fits in memory.
        max_samples (int): If set, cuts the number of samples per task to `max_samples`.
            Note: This should only be used for debugging purposes!
        job_id (int): If the evaluation suite is launched as a slurm job, stores the current job id.
            Purely informative parameter used to retrieve scheduler logs.
        start_time (float): Start time of the experiment. Logged at class init.
        end_time (float): Start time of the experiment. Logged when calling [`GeneralConfigTracker.log_end_time`]
        total_evaluation_time_secondes (str): Inferred total evaluation time in seconds (from the start and end times).
        model_name (str): Name of the currently evaluated model.
        model_sha (str): Commit hash of the currently evaluated model on the hub if available.
        model_dtype (str): Dtype of the model weights, as obtained when loading the model config.
        model_size (str): Model size as obtained when loading the model config.

    """

    # general
    lighteval_sha: str = None
    model_source: str = None
    model_args: str = None
    model_id: str = None
    batch_size: int = None
    max_batch_size: int = None
    max_samples: int = None
    job_id: int = None
    start_time: float = None
    end_time: float = None
    total_evaluation_time_secondes: str = None

    # # model info
    # model_name: str = None
    # model_sha: str = None
    # model_dtype: str = None
    # model_size: str = None

    def __init__(self) -> None:
        """Stores the current lighteval commit for reproducibility, and starts the evaluation timer."""
        repo = git.Repo(os.path.dirname(__file__).split("lm_eval")[0])
        self.lighteval_sha = repo.git.rev_parse("HEAD")
        self.start_time = time.perf_counter()

    # TODO - implement model info for each model
    def _get_model_name(model_args: str) -> str:
        if "peft=" in model_args:
            args_after_peft = model_args.split("peft=")[1]
            return args_after_peft.split(",")[0]
        if "delta=" in model_args:
            args_after_delta = model_args.split("delta=")[1]
            return args_after_delta.split(",")[0]
        if "pretrained=" in model_args:
            args_after_pretrained = model_args.split("pretrained=")[1]
            return args_after_pretrained.split(",")[0]
        else:
            return None

    def log_experiment_args(
        self,
        model_source: str,
        model_args: str,
        batch_size: str,
        max_batch_size: Optional[str] = None,
        max_samples: Optional[Union[int, float]] = None,
        job_id: Optional[str] = None,
    ) -> None:
        """Logs the evaluation parameters."""
        self.model_source = model_source
        self.model_args = model_args
        self.model_name = GeneralConfigTracker._get_model_name(model_args)
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.max_samples = max_samples
        self.job_id = job_id

    # def log_model_info(self, model_info: ModelInfo) -> None:
    #     self.model_name = model_info.model_name
    #     self.model_sha = model_info.model_sha
    #     self.model_dtype = model_info.model_dtype
    #     self.model_size = model_info.model_size

    def log_end_time(self) -> None:
        self.end_time = time.perf_counter()
        self.total_evaluation_time_secondes = str(self.end_time - self.start_time)


class VersionsTracker:
    """Tracker of the tasks versions.

    Tasks can have a version number/date, which indicates what is the precise metric definition and dataset version used for an evaluation.

    Attributes:
        version (dict[str, int]): Maps the task names with the task versions.
    """

    # the versions dict will be a dict of task_name: task_version
    # {"winogrande|winogrande_xl": 0}
    versions: dict[str, int] = {"all": 0}

    def log(self, task_name: str, task_version: int) -> None:
        self.versions[task_name] = task_version
