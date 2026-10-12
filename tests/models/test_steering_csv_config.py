import sys
from types import SimpleNamespace

import pytest
import torch

from lm_eval.models.hf_steered import SteeredModel
from lm_eval.models.huggingface import HFLM


@pytest.mark.parametrize("loader", ["sparsify", "sae_lens"])
@pytest.mark.parametrize("action", ["add", "clamp"])
def test_csv_steering_bias_and_hook(tmp_path, monkeypatch, loader, action):
    vector = torch.tensor([1.0, 2.0, 3.0])
    bias = torch.tensor([0.5, 1.0, 1.5])
    coder = SimpleNamespace(
        W_dec=torch.stack([torch.zeros_like(vector), vector]),
        b_dec=bias,
        cfg=SimpleNamespace(hook_name="layers.0"),
    )
    monkeypatch.setitem(
        sys.modules,
        "sparsify",
        SimpleNamespace(SparseCoder=SimpleNamespace(load_from_hub=lambda *args: coder)),
    )
    monkeypatch.setitem(
        sys.modules,
        "sae_lens",
        SimpleNamespace(SAE=SimpleNamespace(from_pretrained=lambda *args: (coder,))),
    )
    csv_path = tmp_path / "steer.csv"
    csv_path.write_text(
        "loader,action,sparse_model,hookpoint,feature_index,steering_coefficient,sae_id\n"
        f"{loader},{action},example/sae,layers.0,1,2.0,example-id\n"
    )

    config = SteeredModel.derive_steer_config(str(csv_path))["layers.0"]
    torch.testing.assert_close(config["steering_vector"], vector)
    if action == "add":
        assert config["bias"] is None
    else:
        assert config["bias"] is bias

    def init_cpu_model(self, **kwargs):
        self._device = torch.device("cpu")
        self._model = SimpleNamespace(dtype=torch.float32)

    monkeypatch.setattr(HFLM, "__init__", init_cpu_model)
    model = SteeredModel(pretrained="unused", steer_path=str(csv_path), device="cpu")
    acts = torch.ones(1, 2, 3)
    result = model.hook_to_steer["layers.0"](acts)
    if action == "add":
        expected = acts + 2.0 * vector
    else:
        direction = vector / torch.norm(vector)
        centered = acts - bias
        projection = (centered * direction).sum(dim=-1, keepdim=True)
        expected = centered + direction * (2.0 - projection) + bias
    torch.testing.assert_close(result, expected)
