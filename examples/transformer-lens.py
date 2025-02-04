import warnings

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformers import AutoConfig

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def evaluate_lm_eval(lens_model: HookedTransformer, tasks: list[str], **kwargs):
    class HFLikeModelAdapter(nn.Module):
        """Adapts HookedTransformer to match the HuggingFace interface expected by lm-eval"""

        def __init__(self, model: HookedTransformer):
            super().__init__()
            self.model = model
            self.tokenizer = model.tokenizer
            self.config = AutoConfig.from_pretrained(model.cfg.tokenizer_name)
            self.device = model.cfg.device
            self.tie_weights = lambda: self

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            output = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            # Make sure output has the expected .logits attribute
            if not hasattr(output, "logits"):
                if isinstance(output, torch.Tensor):
                    output.logits = output
            return output

        # Only delegate specific attributes we know we need
        def to(self, *args, **kwargs):
            return self.model.to(*args, **kwargs)

        def eval(self):
            self.model.eval()
            return self

        def train(self, mode=True):
            self.model.train(mode)
            return self

    model = HFLikeModelAdapter(lens_model)
    warnings.filterwarnings("ignore", message="Failed to get model SHA for")
    results = evaluator.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=model.tokenizer),
        tasks=tasks,
        verbosity="WARNING",
        **kwargs,
    )
    return results


if __name__ == "__main__":
    # Load base model
    model = HookedTransformer.from_pretrained("pythia-70m")
    res = evaluate_lm_eval(model, tasks=["arc_easy"])
    print(res["results"])
