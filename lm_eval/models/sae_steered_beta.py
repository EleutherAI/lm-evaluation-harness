from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
import torch
from jaxtyping import Float, Int
from typing import List
from torch import Tensor
from functools import partial
from transformer_lens import loading_from_pretrained
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE, HookedSAETransformer
# gemmascope_sae_release = "gemma-scope-2b-pt-res-canonical"
gemmascope_sae_id = "layer_20/width_16k/canonical"

def steering_hook(
    activations: Float[Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + steering_coefficient * sae.W_dec[latent_idx]

class InterventionModel(HookedSAETransformer):  # Replace with the specific model class
    def __init__(self, base_name:str, fwd_hooks:list, device:str='cuda:0'):
        # config = AutoConfig.from_pretrained("google/gemma-2b")
        trueconfig = loading_from_pretrained.get_pretrained_model_config(base_name, device=device)
        super().__init__(trueconfig)
        self.model = HookedSAETransformer.from_pretrained(base_name, device=device)
        self.model.eval()
        self.fwd_hooks = fwd_hooks
        self.device = device  # Add device attribute
        self.to(device)  # Ensure model is on the correct device
    @classmethod
    def from_csv(cls, csv_path: str, base_name:str, device: str = 'cuda:0') -> 'InterventionModel':
        """
        Create an InterventionModel from a CSV file containing steering configurations.
        
        Expected CSV format:
        index, coefficient, sae_release, sae_id, description
        12082, 240.0,gemma-scope-2b-pt-res-canonical,layer_20/width_16k/canonical, increase dogs
        ...

        Args:
            csv_path: Path to the CSV file containing steering configurations
            device: Device to place the model on

        Returns:
            InterventionModel with configured steering hooks
        """
        import pandas as pd
        
        # Read steering configurations
        df = pd.read_csv(csv_path)
        # Create hooks for each row in the CSV
        hooks = []
        for _, row in df.iterrows():
            sae = SAE.from_pretrained(row['sae_release'], row['sae_id'], device=str(device))[0]
            sae.eval()
            hook = partial(
                steering_hook,
                sae=sae,
                latent_idx=int(row['latent_idx']),
                steering_coefficient=float(row['steering_coefficient'])
            )
            hooks.append((sae.cfg.hook_name, hook))
        
        # Create and return the model
        return cls(fwd_hooks=hooks, base_name=base_name, device=device)
    def forward(self, *args, **kwargs):
        # Handle both input_ids and direct tensor inputs
        if 'input_ids' in kwargs:
            input_tensor = kwargs.pop('input_ids')  # Use pop to remove it
        elif args:
            input_tensor = args[0]
            args = args[1:]  # Remove the first argument
        else:
            input_tensor = None
        with torch.no_grad(): # I don't know why this no grad is necessary; I tried putting everything into eval mode. And yet, this is necessary to prevent CUDA out of memory exceptions.
            with self.model.hooks(fwd_hooks=self.fwd_hooks):
                output = self.model.forward(input_tensor, *args, **kwargs)
        return output


@register_model("sae_steered_beta")
class InterventionModelLM(HFLM):
    def __init__(self, base_name, csv_path, **kwargs):
        self.swap_in_model = InterventionModel.from_csv(csv_path=csv_path, base_name=base_name, device=kwargs.get('device', 'cuda'))
        self.swap_in_model.eval()
        # Initialize other necessary attributes
        super().__init__(pretrained=base_name, **kwargs)
        if hasattr(self, '_model'):
            # Delete all the model's parameters but keep the object
            for param in self._model.parameters():
                param.data.zero_()
                param.requires_grad = False
            # Remove all model modules while keeping the base object
            for name, module in list(self._model.named_children()):
                delattr(self._model, name)
            torch.cuda.empty_cache()
            
    
    def _model_call(self, inputs):
        # Implement this method to use your model's forward function
        # import pdb;pdb.set_trace()
        return self.swap_in_model.forward(inputs)#