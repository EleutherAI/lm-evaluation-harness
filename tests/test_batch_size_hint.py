import pytest

torch = pytest.importorskip("torch")  # noqa: F401
transformers = pytest.importorskip("transformers")  # noqa: F401

from lm_eval.models.huggingface import _batch_size_hint_needed


@pytest.mark.parametrize(
    "bs,device,expected",
    [
        (1, "cuda:0", True),        # the silent default, on GPU
        (1, "cuda", True),
        (1, "mps", True),
        (8, "cuda:0", False),       # user already batched
        ("auto", "cuda:0", False),
        (1, "cpu", False),          # CPU: batching pays far less; don't nag
    ],
)
def test_batch_size_hint(bs, device, expected):
    assert _batch_size_hint_needed(bs, device) is expected
