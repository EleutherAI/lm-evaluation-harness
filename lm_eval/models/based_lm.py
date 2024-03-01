import re
from transformers import AutoTokenizer
import torch

from based.utils.hf import load_config_hf

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("based_lm")
class BasedLMWrapper(HFLM):
    def __init__(
            self, 
            checkpoint_name: str='hazyresearch/based-1.3b', 
            arch: str=None,
            device: str = "cuda",
            **kwargs
        ) -> None:

        if arch is None:
            arch = checkpoint_name.split("/")[1].split("-")[0]

        assert arch in ['based', 'mamba', 'attn'], print("`arch` must be one of 'based', 'mamba', or 'attn'")

        if "backend" in kwargs:
            # based currently only supports causal models
            assert kwargs["backend"] == "causal"

        self.checkpoint_name = checkpoint_name

        if arch == "based":
            from based.models.gpt import GPTLMHeadModel
            model = GPTLMHeadModel.from_pretrained_hf(pretrained_model_name=self.checkpoint_name, device=device)
        elif arch == "mamba": 
            from based.models.mamba import MambaLMHeadModel
            model = MambaLMHeadModel.from_pretrained_hf(pretrained_model_name=self.checkpoint_name, device=device)
        elif arch == "attn":
            from based.models.transformer.gpt import GPTLMHeadModel, GPT2Config, state_dict_from_pretrained; # TODO: construct a loading function
            # from based.models.gpt import GPTLMHeadModel
            config_data = load_config_hf(self.checkpoint_name)
            config = GPT2Config(**config_data)
            model = GPTLMHeadModel(config=config, device=device, dtype=torch.float16)
            state_dict = state_dict_from_pretrained(self.checkpoint_name, dtype=torch.float16)
            # remove the 'model.' prefix from the keys
            state_dict = {re.sub("^model\.", "", k): v for k, v in state_dict.items()}
            # remove Unexpected key(s) in state_dict: "train_metrics.num-tokens.count", "val_metrics.num-tokens.count", "test_metrics.num-tokens.count". from the state_dict
            state_dict = {k: v for k, v in state_dict.items() if "metrics" not in k}
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported model {arch}")

        tokenizer_name = kwargs.get("tokenizer", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        model.device = device

        super().__init__(
            pretrained=model,
            # set appropriate defaults for tokenizer, max length, etc
            backend=kwargs.get("backend", "causal"),
            max_length=kwargs.get("max_length", 2048),
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )