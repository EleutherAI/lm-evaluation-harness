import tempfile
from pathlib import Path

import pytest

from lm_eval.tasks import TaskManager


@pytest.fixture(scope="module")
def custom_task_name():
    return "zzz_my_python_task"


@pytest.fixture(scope="module")
def custom_task_tag():
    return "zzz-tag"


@pytest.fixture(scope="module")
def task_yaml(pytestconfig, custom_task_name, custom_task_tag):
    yield f"""include: {pytestconfig.rootpath}/lm_eval/tasks/hellaswag/hellaswag.yaml
task: {custom_task_name}
class: !function {custom_task_name}.MockPythonTask
tag:
  - {custom_task_tag}
"""


@pytest.fixture(scope="module")
def task_code():
    return """
from lm_eval.tasks import ConfigurableTask

class MockPythonTask(ConfigurableTask):

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        config.pop("class")
        super().__init__(data_dir, cache_dir, download_mode, config)
"""


@pytest.fixture(scope="module")
def custom_task_files_dir(task_yaml, task_code, custom_task_name):
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = Path(temp_dir) / f"{custom_task_name}.yaml"
        with open(yaml_path, "w") as f:
            f.write(task_yaml)
        pysource_path = Path(temp_dir) / f"{custom_task_name}.py"
        with open(pysource_path, "w") as f:
            f.write(task_code)
        yield temp_dir


def test_python_task_inclusion(
    custom_task_files_dir: Path, custom_task_name: str, custom_task_tag: str
):
    task_manager = TaskManager(
        verbosity="INFO", include_path=str(custom_task_files_dir)
    )
    # check if python tasks enters the global task_index
    assert custom_task_name in task_manager.task_index
    # check if subtask is present
    assert custom_task_name in task_manager.all_subtasks
    # check if tag is present
    assert custom_task_tag in task_manager.all_tags
    # check if it can be loaded by tag (custom_task_tag)
    assert custom_task_name in task_manager.load_task_or_group(custom_task_tag)
