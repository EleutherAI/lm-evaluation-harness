import importlib
from pathlib import Path

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml


@pytest.fixture(
    params=[
        ("lm_eval.tasks.ifeval.utils", "", "ifeval.yaml"),
        ("lm_eval.tasks.leaderboard.ifeval.utils", "", "ifeval.yaml"),
        ("lm_eval.tasks.ifeval.multilingual.utils", "es:", "ifeval_es.yaml"),
        ("lm_eval.tasks.ifeval.multilingual.utils", "ca:", "ifeval_ca.yaml"),
    ],
    ids=["ifeval", "leaderboard", "spanish", "catalan"],
)
def variant(request):
    pytest.importorskip("langdetect")
    pytest.importorskip("immutabledict")
    module, prefix, config_file = request.param
    if prefix:
        # These checkers import both language models while loading their registry.
        for dependency in ("spacy", "iso639", "es_core_news_sm", "ca_core_news_sm"):
            pytest.importorskip(dependency)
    return importlib.import_module(module), prefix, config_file


@pytest.mark.parametrize("mode", ["strict", "loose"])
@pytest.mark.parametrize(
    "instruction,kwargs,response",
    [
        (
            "detectable_content:number_placeholders",
            {"num_placeholders": 0},
            "A plain response.",
        ),
        (
            "detectable_format:number_bullet_lists",
            {"num_bullets": 0},
            "A plain response.",
        ),
        (
            "detectable_format:number_highlighted_sections",
            {"num_highlights": 0},
            "A plain response.",
        ),
        (
            "detectable_content:number_placeholders",
            {"num_placeholders": 1},
            "Hello [name].",
        ),
        (
            "detectable_format:number_bullet_lists",
            {"num_bullets": 2},
            "- first\n- second",
        ),
    ],
)
def test_instruction_arguments_preserve_zero_counts(
    variant, mode, instruction, kwargs, response
):
    utils, prefix, _ = variant
    example = utils.InputExample(
        key=1,
        instruction_id_list=[prefix + instruction],
        prompt="Follow the supplied count.",
        kwargs=[{**kwargs, "unused_dataset_field": None}],
    )

    result = getattr(utils, f"test_instruction_following_{mode}")(example, response)

    assert result.follow_all_instructions
    assert result.follow_instruction_list == [True]


@pytest.mark.parametrize("mode", ["strict", "loose"])
def test_zero_counts_do_not_allow_an_empty_response(variant, mode):
    utils, prefix, _ = variant
    example = utils.InputExample(
        key=1,
        instruction_id_list=[prefix + "detectable_format:number_bullet_lists"],
        prompt="Do not include bullet points.",
        kwargs=[{"num_bullets": 0}],
    )

    result = getattr(utils, f"test_instruction_following_{mode}")(example, "")

    assert not result.follow_all_instructions
    assert result.follow_instruction_list == [False]


def test_configured_ifeval_task_preserves_zero_count_requirements(variant):
    utils, prefix, config_file = variant
    doc = {
        "key": 1,
        "prompt": "Do not use bullet points. No placeholders are required.",
        "instruction_id_list": [
            prefix + "detectable_format:number_bullet_lists",
            prefix + "detectable_content:number_placeholders",
        ],
        "kwargs": [{"num_bullets": 0}, {"num_placeholders": 0}],
    }

    class LocalIFEvalTask(ConfigurableTask):
        def download(self, dataset_kwargs=None, **kwargs):
            self.dataset = datasets.DatasetDict(
                train=datasets.Dataset.from_list([doc]),
                test=datasets.Dataset.from_list([doc]),
            )

    task = LocalIFEvalTask(
        config=load_yaml(Path(utils.__file__).parent / config_file, resolve_func=True)
    )
    request = task.construct_requests(
        doc, task.doc_to_text(doc), metadata=(task.config.task, 0, 1)
    )
    assert request.request_type == "generate_until"

    scores = task.process_results(doc, ["A plain response."])

    assert scores == {
        "prompt_level_strict_acc": True,
        "inst_level_strict_acc": [True, True],
        "prompt_level_loose_acc": True,
        "inst_level_loose_acc": [True, True],
    }
