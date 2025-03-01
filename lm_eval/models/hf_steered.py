from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Union

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
    def __init__(
        self,
        pretrained: str,
        steer_path: str,
        device: Union[str, None] = None,
        **kwargs,
    ):
        """
        HFLM with a steered forward pass.

        To derive steering vectors from a sparse model loadable with sparsify or sae_lens,
        provide the path to a CSV file with the following columns (example rows are provided below):

        loader,action,sparse_model,hookpoint,feature_index,steering_coefficient,sae_id,description,
        sparsify,add,EleutherAI/sae-pythia-70m-32k,layers.3,30,10.0,,,
        sae_lens,add,gemma-scope-2b-pt-res-canonical,layers.20,12082,240.0,layer_20/width_16k/canonical,increase dogs,

        To load steering vectors directly, provide the path to a pytorch (.pt) file with content in the following format:

        [
            {
                "hookpoint": <str>,
                "steering_vector": <torch.Tensor>,
                "steering_coefficient": <float>,
                "action": <Literal["add", "clamp_max", "clamp_constant"]>,
            },
            ...
        ]
        """
        super().__init__(pretrained=pretrained, device=device, **kwargs)

        if steer_path.endswith(".pt") or steer_path.endswith(".pth"):
            with open(steer_path, "rb") as f:
                steer_config = torch.load(f)
        elif steer_path.endswith(".csv"):
            steer_config = self.derive_steer_config(steer_path)
        else:
            raise ValueError(f"Unknown steer file type: {steer_path}")

        hook_to_steer = {}
        for hookpoint, steer_info in steer_config.items():
            action = steer_info["action"]
            steering_coefficient = steer_info["steering_coefficient"]
            steering_vector = steer_info["steering_vector"]

            if action == "add":
                # Steers the model by adding some multiple of a steering vector to all sequence positions.
                hook_to_steer[hookpoint] = (
                    lambda acts: acts + steering_coefficient * steering_vector
                )
            elif action == "clamp_max":
                hook_to_steer[hookpoint] = partial(
                    self.ceiling_clamp,
                    steering_vector=steering_vector,
                    value=steering_coefficient,
                )
            elif action == "clamp_constant":
                hook_to_steer[hookpoint] = partial(
                    self.clamp_constant,
                    steering_vector=steering_vector,
                    value=steering_coefficient,
                )
            else:
                raise ValueError(f"Unknown hook type: {action}")

        self.hook_to_steer = hook_to_steer

    @classmethod
    def derive_steer_config(cls, steer_path: str):
        """Derive a dictionary of steering vectors from sparse model(/s) specified in a CSV file."""
        import pandas as pd

        df = pd.read_csv(steer_path)
        steer_data = {}

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

                steering_vector = (
                    sparse_coder.W_dec[feature_index]
                    if sparse_coder.W_dec is not None
                    else sparse_coder.encoder.weight[feature_index]
                )
            elif loader == "sae_lens":
                sparse_coder = load_from_sae_lens(
                    sae_release=sparse_name, sae_id=row["sae_id"]
                )
                steering_vector = sparse_coder.W_dec[feature_index]
                if hookpoint == "" or pd.isna(hookpoint):
                    hookpoint = sparse_coder.cfg.hook_name
            else:
                raise ValueError(f"Unknown loader: {loader}")

            steer_data[hookpoint] = {
                "action": action,
                "steering_coefficient": steering_coefficient,
                "steering_vector": steering_vector,
            }

        return steer_data

    @classmethod
    def ceiling_clamp(
        cls, acts: Tensor, steering_vector: Tensor, value: float
    ) -> Tensor:
        """Clamps a direction of the activations to a fixed maximum norm.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, features]
            steering_vector (Tensor): A direction to clamp of shape [features]
            value (float): Value to clamp the direction to

        Returns:
            Tensor: The modified activations with the specified direction clamped
        """
        norm_direction = steering_vector / steering_vector.norm()
        proj = torch.einsum("bpf,f->bp", acts, norm_direction).unsqueeze(-1)

        # Assign each position a scaling factor to clamp its projection norm
        scaling_factor = torch.zeros_like(proj)
        mask = proj > value
        scaling_factor[mask] = (value - proj[mask]) / proj[mask]

        adjustment = torch.einsum("bp,f->bpf", proj * scaling_factor, norm_direction)
        return acts + adjustment

    @classmethod
    def clamp_constant(cls, acts: Tensor, steering_vector: Tensor, value: float):
        """Clamps a direction of the activations to a fixed constant value.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, features]
            steering_vector (Tensor): A direction to clamp of shape [features]
            value (float): Value to clamp the direction to

        Returns:
            Tensor: The modified activations with the specified direction clamped
        """

        direction = steering_vector / torch.norm(steering_vector)

        proj_magnitude = torch.sum(acts * direction, dim=-1, keepdim=True)
        proj_vector = proj_magnitude * direction

        scale_factor = torch.where(
            proj_magnitude > 0,
            value / torch.abs(proj_magnitude),
            torch.zeros_like(proj_magnitude),
        )
        normalized_proj = proj_vector * scale_factor

        return acts - proj_vector + normalized_proj

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
