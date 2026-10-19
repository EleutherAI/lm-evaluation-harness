from pathlib import Path

import pytest

from lm_eval.tasks._yaml_loader import load_yaml


EVALITA_DIR = Path(__file__).parent.parent / "lm_eval" / "tasks" / "evalita_llm"
NER_V2_YAMLS = sorted(EVALITA_DIR.glob("_evalita-mp_ner-*_group_p*.yaml"))


def test_task_yamls_found():
    assert len(NER_V2_YAMLS) == 6


@pytest.mark.parametrize("path", NER_V2_YAMLS, ids=lambda p: p.stem)
def test_ner_v2_prompts_share_splits_and_target(path):
    config = load_yaml(path, resolve_func=False)

    assert config["test_split"] == "reduced_test"
    assert config["fewshot_split"] == "dev"
    assert str(config["doc_to_target"]).endswith("utils.ner_doc_to_target")
