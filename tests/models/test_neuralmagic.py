import pytest

import lm_eval.evaluator as evaluator
from lm_eval.api.registry import get_model


SPARSEML_MODELS_TASKS = {
    "facebook/opt-125m": "lambada_openai",
    "hf-internal-testing/tiny-random-gpt2": "wikitext",
    "mgoin/llama2.c-stories15M-quant-pt": "wikitext",
}

DEEPSPARSE_MODELS_TASKS = {
    "hf:mgoin/llama2.c-stories15M-quant-ds": "lambada_openai",
}


@pytest.mark.parametrize("model_id,task", SPARSEML_MODELS_TASKS.items())
def test_sparseml_eval(model_id, task):

    lm = get_model("sparseml").create_from_arg_string(
        f"pretrained={model_id}",
        {
            "batch_size": 1,
            "device": "cpu",
        },
    )

    limit = 10
    evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        limit=limit,
        bootstrap_iters=10,
    )


@pytest.mark.parametrize("model_id,task", DEEPSPARSE_MODELS_TASKS.items())
def test_deepsparse_eval(model_id, task):

    lm = get_model("deepsparse").create_from_arg_string(
        f"pretrained={model_id}",
        {
            "batch_size": 1,
        },
    )

    limit = 10
    evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        limit=limit,
        bootstrap_iters=10,
    )
