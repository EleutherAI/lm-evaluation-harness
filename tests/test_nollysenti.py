import importlib

import datasets
import pytest

from lm_eval.tasks._yaml_loader import load_yaml


PROMPTS = [f"prompt_{i}" for i in range(1, 6)]
BASE = "lm_eval/tasks/afrobench/nollysenti/{}/nollysenti"


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("lang", ["en", "ha", "ig", "pcm", "yo"])
def test_nollysenti_process_docs_renames_hub_columns(prompt, lang):
    utils = importlib.import_module(
        f"lm_eval.tasks.afrobench.nollysenti.{prompt}.utils"
    )
    dataset = datasets.Dataset.from_dict(
        {
            f"{lang}_review": ["great film", "boring"],
            "sentiment": ["positive", "negative"],
        }
    )

    docs = utils.process_docs(dataset)

    assert docs.column_names == ["review", "label"]
    assert docs[0] == {"review": "great film", "label": "positive"}


@pytest.mark.parametrize("prompt", PROMPTS)
def test_nollysenti_base_config_uses_process_docs(prompt):
    config = load_yaml(BASE.format(prompt), resolve_func=False)
    assert "process_docs" in config
