import runpy
import shutil
from pathlib import Path

import pytest

from lm_eval.tasks._index import TaskIndex


def test_generated_translation_tasks_remain_discoverable(tmp_path):
    pytest.importorskip("pycountry")
    source = Path(__file__).resolve().parents[1] / "lm_eval/tasks/translation"
    generator = runpy.run_path(str(source / "utils.py"))["gen_lang_yamls"]
    shutil.copy(source / "wmt_common_yaml", tmp_path)
    generator(str(tmp_path), overwrite=False)

    expected = TaskIndex.build([source])
    actual = TaskIndex.build([tmp_path])
    assert actual.keys() == expected.keys()
    for name, entry in expected.items():
        assert actual[name].kind == entry.kind
        assert actual[name].tags == entry.tags
