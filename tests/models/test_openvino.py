import random
import tempfile
from pathlib import Path

import pytest
from optimum.intel import OVModelForCausalLM, OVModelForSeq2SeqLM
from transformers import AutoTokenizer

from lm_eval import evaluator
from lm_eval.api.registry import get_model


SUPPORTED_ARCHITECTURES_TASKS = [
    (
        "causal",
        "facebook/opt-125m",
        "lambada_openai",
    ),
    (
        "causal",
        "hf-internal-testing/tiny-random-gpt2",
        "wikitext",
    ),
    (
        "seq2seq",
        "hf-internal-testing/tiny-random-t5",
        "sst2",
    ),
]


@pytest.mark.parametrize("backend,model_id,task", SUPPORTED_ARCHITECTURES_TASKS)
def test_evaluator(backend, model_id, task):
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_cls = OVModelForCausalLM if backend == "causal" else OVModelForSeq2SeqLM
        model = model_cls.from_pretrained(model_id, export=True, use_cache=True)
        model.save_pretrained(tmpdirname)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(tmpdirname)

        lm = get_model("openvino").create_from_arg_string(
            f"pretrained={tmpdirname},backend={backend}",
            {
                "batch_size": 1,
                "device": "cpu",
            },
        )

        def ll_fn(reqs):
            for ctx, cont in [req.args for req in reqs]:
                if len(ctx) == 0:
                    continue
                # space convention
                assert ctx[-1] != " "
                assert cont[0] == " " or ctx[-1] == "\n"

            res = []

            random.seed(42)
            for _ in reqs:
                res.extend([(-random.random(), False)])

            return res

        def ll_perp_fn(reqs):
            for (string,) in [req.args for req in reqs]:
                assert isinstance(string, str)

            res = []
            random.seed(42)
            for _ in reqs:
                res.extend([-random.random()])

            return res

        lm.loglikelihood = ll_fn
        lm.loglikelihood_rolling = ll_perp_fn

        limit = 10
        evaluator.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=0,
            limit=limit,
            bootstrap_iters=10,
        )


def test_ov_config():
    """Test that if specified, a custom OpenVINO config is loaded correctly"""
    model_id = "hf-internal-testing/tiny-random-gpt2"
    with tempfile.TemporaryDirectory() as tmpdirname:
        config_file = str(Path(tmpdirname) / "ov_config.json")
        with open(Path(config_file), "w", encoding="utf-8") as f:
            f.write('{"DYNAMIC_QUANTIZATION_GROUP_SIZE" : "32"}')
        lm = get_model("openvino").create_from_arg_string(
            f"pretrained={model_id},ov_config={config_file}"
        )
    assert (
        lm.model.request.get_compiled_model().get_property(
            "DYNAMIC_QUANTIZATION_GROUP_SIZE"
        )
        == 32
    )
