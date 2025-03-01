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
        HFLM with a steered forward pass specified in a CSV file.

        Expected CSV format:
        loader,hook_action,sparse_model,hookpoint,feature_index,steering_coefficient,sae_id,description,
        sae_lens,add,gemma-scope-2b-pt-res-canonical,layers.20,12082,10.0,layer_20/width_16k/canonical,increase dogs,
        """
        super().__init__(pretrained=pretrained, device=device, **kwargs)

        # Resolve the interventions specified in the CSV file
        import pandas as pd

        df = pd.read_csv(steer_path)

        if any(df["loader"] == "sparsify"):
            from sparsify import SparseCoder
        if any(df["loader"] == "sae_lens"):
            from sae_lens import SAE

            sae_cache = {}

            def load_from_sae_lens(sae_release: str, sae_id: str):
                cache_key = (sae_release, sae_id)
                if cache_key not in sae_cache:
                    sae = SAE.from_pretrained(sae_release, sae_id, device=str(device))[
                        0
                    ]
                    sae.use_error_term = True
                    sae.eval()

                    sae_cache[cache_key] = sae

                return sae_cache[cache_key]

        hook_to_steer: dict[str, Callable] = {}

        for _, row in df.iterrows():
            loader = row.get("loader", "sparsify")
            hook_action = row.get("hook_action", "add")
            sparse_name = row["sparse_model"]
            hookpoint = row["hookpoint"]
            feature_index = int(row["feature_index"])
            steering_coefficient = float(row["steering_coefficient"])
            if loader == "sae_lens":
                sae_id = row["sae_id"]

            if loader == "sparsify":
                name_path = Path(sparse_name)

                sparse_coder = (
                    SparseCoder.load_from_disk(
                        name_path / hookpoint, device=device or self._model.device
                    )
                    if name_path.exists()
                    else SparseCoder.load_from_hub(
                        sparse_name, hookpoint, device=device or self._model.device
                    )
                )

                steering_vector = (
                    sparse_coder.W_dec[feature_index]
                    if sparse_coder.W_dec is not None
                    else sparse_coder.encoder.weight[feature_index]
                )
            elif loader == "sae_lens":
                sparse_coder = load_from_sae_lens(
                    sae_release=sparse_name, sae_id=sae_id
                )
                steering_vector = sparse_coder.W_dec[feature_index]
                if hookpoint == "" or pd.isna(hookpoint):
                    hookpoint = sparse_coder.cfg.hook_name
            else:
                raise ValueError(f"Unknown loader: {loader}")

            if hook_action == "add":
                # Steers the model by adding some multiple of a steering vector to all sequence positions.
                hook_to_steer[hookpoint] = (
                    lambda acts: acts + steering_coefficient * steering_vector
                )
            elif hook_action == "clamp":
                hook_to_steer[hookpoint] = partial(
                    self.clamp_original,
                    latent_idx=torch.tensor([feature_index]),
                    value=steering_coefficient,
                )
            else:
                raise ValueError(f"Unknown hook type: {hook_action}")

        self.hook_to_steer = hook_to_steer

    @classmethod
    def clamp_original(cls, acts: Tensor, latent_idx: Tensor, value: float) -> Tensor:
        """Clamps a specific latent feature in the sparse activations to a fixed value
        if the current activation is greater than 0.

        Args:
            acts (Tensor): The activations tensor to edit, shape [batch, pos, features]
            latent_idx (Tensor): The activation index to clamp, shape [features]
            value (float): Value to clamp the feature to

        Returns:
            Tensor: The modified sparse activations with the specified feature clamped
        """
        mask = acts[:, :, latent_idx] > 0
        acts[:, :, latent_idx][mask] = value

        return acts

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
