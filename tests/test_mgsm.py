import shutil
from pathlib import Path

import pytest

from lm_eval.tasks import TaskManager
from lm_eval.tasks.mgsm.utils import gen_lang_yamls


MGSM_PATH = Path(__file__).parent.parent / "lm_eval" / "tasks" / "mgsm"
LANGUAGES = ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")
VARIANTS = (
    ("direct", "direct", "direct_yaml", "mgsm_direct", "mgsm_direct"),
    ("en_cot", "en-cot", "cot_yaml", "mgsm_cot_en", "mgsm_en_cot"),
    ("native_cot", "native-cot", "cot_yaml", "mgsm_cot_native", "mgsm_native_cot"),
)


@pytest.fixture(scope="module", params=["checked_in", "regenerated"])
def mgsm_manager(request, tmp_path_factory):
    root = MGSM_PATH
    if request.param == "regenerated":
        root = tmp_path_factory.mktemp("mgsm")
        for directory, mode, template, _, _ in VARIANTS:
            output = root / directory
            output.mkdir()
            shutil.copyfile(MGSM_PATH / directory / template, output / template)
            gen_lang_yamls(str(output), overwrite=False, mode=mode)
    return TaskManager(include_path=root, include_defaults=False)


@pytest.mark.parametrize("tag,prefix", [(v[3], v[4]) for v in VARIANTS])
def test_mgsm_tag_membership(mgsm_manager, tag, prefix):
    """Each variant selects only its own eleven existing language task names."""
    expected = {f"{prefix}_{language}" for language in LANGUAGES}

    assert tag in mgsm_manager.all_tags
    assert mgsm_manager.match_tasks([tag]) == [tag]
    assert mgsm_manager.task_index[tag].tags == expected
    assert expected <= set(mgsm_manager.all_subtasks)
