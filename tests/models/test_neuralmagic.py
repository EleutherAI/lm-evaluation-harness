import pytest

from lm_eval import evaluator
from lm_eval.api.registry import get_model


SPARSEML_MODELS_TASKS = [
    # loglikelihood
    ("facebook/opt-125m", "lambada_openai"),
    # loglikelihood_rolling
    ("hf-internal-testing/tiny-random-gpt2", "wikitext"),
    # generate_until
    ("mgoin/tiny-random-llama-2-quant", "gsm8k"),
]

DEEPSPARSE_MODELS_TASKS = [
    # loglikelihood
    ("hf:mgoin/llama2.c-stories15M-quant-ds", "lambada_openai"),
    # loglikelihood_rolling (not supported yet)
    # ("hf:mgoin/llama2.c-stories15M-quant-ds", "wikitext"),
    # generate_until
    ("hf:mgoin/llama2.c-stories15M-quant-ds", "gsm8k"),
]


@pytest.mark.skip(reason="test failing")
@pytest.mark.parametrize("model_id,task", SPARSEML_MODELS_TASKS)
def test_sparseml_eval(model_id, task):
    lm = get_model("sparseml").create_from_arg_string(
        f"pretrained={model_id}",
        {
            "batch_size": 1,
            "device": "cpu",
            "dtype": "float32",
        },
    )

    limit = 5
    evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        limit=limit,
    )


@pytest.mark.parametrize("model_id,task", DEEPSPARSE_MODELS_TASKS)
def test_deepsparse_eval(model_id, task):
    lm = get_model("deepsparse").create_from_arg_string(
        f"pretrained={model_id}",
        {
            "batch_size": 1,
        },
    )

    limit = 5
    evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        limit=limit,
    )
