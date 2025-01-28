"""Credit: contributed by https://github.com/AMindToThink aka Matthew Khoriaty of Northwestern University."""



from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.InterventionModel import InterventionModel

import torch

@register_model("sae_steered_beta")
class InterventionModelLM(HFLM):
    def __init__(self, base_name, csv_path, **kwargs):
        self.swap_in_model = InterventionModel.from_csv(
            csv_path=csv_path, base_name=base_name, device=kwargs.get("device", "cuda")
        )
        self.swap_in_model.eval()
        # Initialize other necessary attributes
        super().__init__(pretrained=base_name, **kwargs)
        if hasattr(self, "_model"):
            # Delete all the model's parameters but keep the object
            for param in self._model.parameters():
                param.data.zero_()
                param.requires_grad = False
            # Remove all model modules while keeping the base object
            for name, module in list(self._model.named_children()):
                delattr(self._model, name)
            torch.cuda.empty_cache()

    def _model_call(self, inputs):
        return self.swap_in_model.forward(inputs)
