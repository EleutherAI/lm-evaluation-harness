import os
import tempfile
import lm_eval.base as base
import lm_eval.tasks as tasks
import lm_eval.models as models
import lm_eval.evaluator as evaluator
import random
import pytest

from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM

SUPPORTED_ARCHITECTURES_TASKS = {
        "facebook/opt-125m": "lambada_openai",
        "hf-internal-testing/tiny-random-gpt2": "wikitext"
}

@pytest.mark.parametrize("model_id,task", SUPPORTED_ARCHITECTURES_TASKS.items())
def test_evaluator(model_id, task):
    with tempfile.TemporaryDirectory() as tmpdirname:
        model = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True)
        model.save_pretrained(tmpdirname)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(tmpdirname)
        
    
        lm = models.get_model("optimum-causal").create_from_arg_string(
                f"pretrained={tmpdirname}",
                {
                    "batch_size": 1,
                    "device": "cpu",
                },
            )
        
        task_dict = tasks.get_task_dict([task])

        def ll_fn(reqs):
            for ctx, cont in reqs:
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
            for (string,) in reqs:
                assert isinstance(string, str)

            res = []
            random.seed(42)
            for _ in reqs:
                res.append(-random.random())

            return res

        lm.loglikelihood = ll_fn
        lm.loglikelihood_rolling = ll_perp_fn

        limit = 10
        evaluator.evaluate(
            lm=lm,
            task_dict=task_dict,
            num_fewshot=0,
            limit=limit,
            bootstrap_iters=10,
            description_dict=None,
        )
    