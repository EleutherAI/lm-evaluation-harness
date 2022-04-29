import random
import lm_eval.models as models
import pytest
import torch
from transformers import StoppingCriteria


@pytest.mark.parametrize(
    "eos_token,test_input,expected", 
    [
        ("not", "i like", "i like to say that I'm not"), 
        ("say that", "i like", "i like to say that"),
        ("great", "big science is", "big science is a great"),
        ("<|endoftext|>", "big science has", "big science has been done in the past, but it's not the same as the science of the")
    ]
)
def test_stopping_criteria(eos_token, test_input, expected):
    random.seed(42)
    torch.random.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2 = models.get_model("gpt2")(device=device)

    context = torch.tensor([gpt2.tokenizer.encode(test_input)])
    stopping_criteria_ids = gpt2.tokenizer.encode(eos_token)

    generations = gpt2._model_generate(
        context,
        max_length=20,
        stopping_criteria_ids=stopping_criteria_ids
    )
    generations = gpt2.tokenizer.decode(generations[0])
    assert generations == expected
