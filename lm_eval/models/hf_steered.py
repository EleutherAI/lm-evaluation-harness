from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

import torch
from peft.peft_model import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedModel

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@contextmanager
def steer(
    model: Union[PreTrainedModel, PeftModel], hook_to_steer: dict[str, Callable]
) -> Generator[None, Any, None]:
    """
    Context manager that temporarily hooks models and steers them.

    Args:
        model: The transformer model to hook
        hook_to_steer: Dictionary mapping hookpoints to steering functions

    Yields:
        None
    """

    def create_hook(hookpoint: str):
        def hook_fn(module: nn.Module, input: Any, output: Tensor):
            # If output is a tuple (like in some transformer layers), take first element
            if isinstance(output, tuple):
                output = (hook_to_steer[hookpoint](output[0]), *output[1:])  # type: ignore
            else:
                output = hook_to_steer[hookpoint](output)

            return output

        return hook_fn

    handles = []
    hookpoints = list(hook_to_steer.keys())

    for name, module in model.base_model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_hook(name))
            handles.append(handle)

    if len(handles) != len(hookpoints):
        raise ValueError(f"Not all hookpoints could be resolved: {hookpoints}")

    try:
        yield None
    finally:
        for handle in handles:
            handle.remove()


@register_model("steered")
class SteeredModel(HFLM):
    hook_to_steer: dict[str, Callable]

    def __init__(
        self,
        pretrained: str,
        steer_path: str,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        HFLM with a steered forward pass.

        To load steering vectors directly, provide the path to a pytorch (.pt) file with content in the following format:

        {
            hookpoint: {
                "steering_vector": <torch.Tensor>,
                "steering_coefficient": <float>,
                "action": <Literal["add", "clamp"]>,
                "bias": <torch.Tensor | None>,
                "head_index": <int | None>,
            },
            ...
        }

        To derive steering vectors from a sparse model loadable with sparsify or sae_lens,
        provide the path to a CSV file with the following columns (example rows are provided below):

        loader,action,sparse_model,hookpoint,feature_index,steering_coefficient,head_index,sae_id,description,
        sparsify,add,EleutherAI/sae-pythia-70m-32k,layers.3,30,10.0,,,,
        sae_lens,add,gemma-scope-2b-pt-res-canonical,layers.20,12082,240.0,,layer_20/width_16k/canonical,increase dogs,
        """
        super().__init__(pretrained=pretrained, device=device, **kwargs)

        if steer_path.endswith(".pt") or steer_path.endswith(".pth"):
            with open(steer_path, "rb") as f:
                steer_config: dict[str, dict[str, Any]] = torch.load(
                    f, weights_only=True
                )
        elif steer_path.endswith(".csv"):
            steer_config = self.derive_steer_config(steer_path)
        else:
            raise ValueError(f"Unknown steer file type: {steer_path}")

        hook_to_steer = {}
        for hookpoint, steer_info in steer_config.items():
            action = steer_info["action"]
            steering_vector = (
                steer_info["steering_vector"].to(self.device).to(self.model.dtype)
            )
            steering_coefficient = float(steer_info.get("steering_coefficient", 1.0))
            head_index = steer_info.get("head_index", None)
            bias = steer_info.get("bias", None)
            if bias is not None:
                bias = bias.to(self.device).to(self.model.dtype)

            if action == "add":
                # Steer the model by adding a multiple of a steering vector to all sequence positions.
                assert bias is None, "Bias is not supported for the `add` action."
                hook_to_steer[hookpoint] = partial(
                    self.add,
                    vector=steering_vector * steering_coefficient,
                    head_index=head_index,
                )
            elif action == "clamp":
                # Steer the model by clamping the activations to a value in the direction of the steering vector.
                hook_to_steer[hookpoint] = partial(
                    self.clamp,
                    direction=steering_vector / torch.norm(steering_vector),
                    value=steering_coefficient,
                    bias=bias,
                    head_index=head_index,
                )
            else:
                raise ValueError(f"Unknown hook type: {action}")

        self.hook_to_steer = hook_to_steer

    @classmethod
    def derive_steer_config(cls, steer_path: str):
        """Derive a dictionary of steering vectors from sparse model(/s) specified in a CSV file."""
        import pandas as pd

        df = pd.read_csv(steer_path)
        steer_data: dict[str, dict[str, Any]] = {}

        if any(df["loader"] == "sparsify"):
            from sparsify import SparseCoder
        if any(df["loader"] == "sae_lens"):
            from sae_lens import SAE

            sae_cache = {}

            def load_from_sae_lens(sae_release: str, sae_id: str):
                cache_key = (sae_release, sae_id)
                if cache_key not in sae_cache:
                    sae_cache[cache_key] = SAE.from_pretrained(sae_release, sae_id)[0]

                return sae_cache[cache_key]

        for _, row in df.iterrows():
            action = row.get("action", "add")
            sparse_name = row["sparse_model"]
            hookpoint = row["hookpoint"]
            feature_index = int(row["feature_index"])
            steering_coefficient = float(row["steering_coefficient"])
            loader = row.get("loader", "sparsify")

            if loader == "sparsify":
                name_path = Path(sparse_name)

                sparse_coder = (
                    SparseCoder.load_from_disk(name_path / hookpoint)
                    if name_path.exists()
                    else SparseCoder.load_from_hub(sparse_name, hookpoint)
                )
                assert sparse_coder.W_dec is not None

                steering_vector = sparse_coder.W_dec[feature_index]
                bias = sparse_coder.b_dec

            elif loader == "sae_lens":
                sparse_coder = load_from_sae_lens(
                    sae_release=sparse_name, sae_id=row["sae_id"]
                )
                steering_vector = sparse_coder.W_dec[feature_index]
                bias = sparse_coder.b_dec
                if hookpoint == "" or pd.isna(hookpoint):
                    hookpoint = sparse_coder.cfg.hook_name
            else:
                raise ValueError(f"Unknown loader: {loader}")

            steer_data[hookpoint] = {
                "action": action,
                "steering_coefficient": steering_coefficient,
                "steering_vector": steering_vector,
                "bias": bias,
            }

        return steer_data

    @classmethod
    def add(
        cls,
        acts: Tensor,
        vector: Tensor,
        head_index: Optional[int],
    ):
        """Adds the given vector to the activations.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, ..., features]
            vector (Tensor): A vector to add of shape [features]
            head_index (int | None): Optional attention head index to add to
        """
        if head_index is not None:
            acts[:, :, head_index, :] = acts[:, :, head_index, :] + vector
        else:
            acts = acts + vector

        return acts

    @classmethod
    def clamp(
        cls,
        acts: Tensor,
        direction: Tensor,
        value: float,
        head_index: Optional[int],
        bias: Optional[Tensor] = None,
    ):
        """Clamps the activations to a given value in a specified direction. The direction
        must be a unit vector.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, ..., features]
            direction (Tensor): A direction to clamp of shape [features]
            value (float): Value to clamp the direction to
            head_index (int | None): Optional attention head index to clamp
            bias (Tensor | None): Optional bias to add to the activations

        Returns:
            Tensor: The modified activations with the specified direction clamped
        """
        if bias is not None:
            acts = acts - bias

        if head_index is not None:
            x = acts[:, :, head_index, :]
            proj = (x * direction).sum(dim=-1, keepdim=True)
            assert proj == acts @ direction

            clamped = acts.clone()
            clamped[:, :, head_index, :] = x + direction * (value - proj)
        else:
            proj = torch.sum(acts * direction, dim=-1, keepdim=True)
            clamped = acts + direction * (value - proj)

        if bias is not None:
            return clamped + bias

        return clamped

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            with steer(self.model, self.hook_to_steer):
                return self.model.forward(*args, **kwargs)

    def _model_call(self, *args, **kwargs):
        with steer(self.model, self.hook_to_steer):
            return super()._model_call(*args, **kwargs)

    def _model_generate(self, *args, **kwargs):
        with steer(self.model, self.hook_to_steer):
            return super()._model_generate(*args, **kwargs)
