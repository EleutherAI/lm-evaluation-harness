import os
import multiprocessing

from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
import intel_extension_for_pytorch as ipex

import torch
from typing import Optional, Union
from lm_eval.base import BaseLM

from transformers import AutoTokenizer, LlamaTokenizer

def _get_dtype(
    dtype: Union[str, torch.dtype]
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class ChatGLMGPULM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int,str))
        
        model = AutoModelForCausalLM.from_pretrained(pretrained,
                                          load_in_4bit=True,
                                          optimize_model=True,
                                          trust_remote_code=True,
                                          use_cache=True)

        self.model = model.to('xpu')

        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained, trust_remote_code=True)

        # setup for automatic batch size detection
        if batch_size == 'auto':
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.model.token_eos()

    @property
    def max_length(self):
        return 2048  # TODO: how to get this from config

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return torch.device("cpu")

    def tok_encode(self, string: str):
        input_ids = self.tokenizer.encode(string)
        return input_ids

    def tok_decode(self, tokens):
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.inference_mode():
            inps = inps.to('xpu')
            res = self.model(inps)[0]
            return res

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model(context, max_tokens=max_length, stop=["Q:", "\n"], echo=True)