import random
import tempfile

import pytest
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

import lm_eval.evaluator as evaluator
from lm_eval.api.registry import get_model


SUPPORTED_ARCHITECTURES_TASKS = {
    "facebook/opt-125m": "lambada_openai",
    "hf-internal-testing/tiny-random-gpt2": "wikitext",
}


@pytest.mark.parametrize("model_id,task", SUPPORTED_ARCHITECTURES_TASKS.items())
def test_evaluator(model_id, task):
    with tempfile.TemporaryDirectory() as tmpdirname:
        model = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=True
        )
        model.save_pretrained(tmpdirname)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(tmpdirname)

        lm = get_model("openvino").create_from_arg_string(
            f"pretrained={tmpdirname}",
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
                res.append((-random.random(), False))

            return res

        def ll_perp_fn(reqs):
            for (string,) in [req.args for req in reqs]:
                assert isinstance(string, str)

            res = []
            random.seed(42)
            for _ in reqs:
                res.append(-random.random())

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
