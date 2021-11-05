import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_eval.base import LM, BaseLM
from lm_eval import utils
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterable


class HFLM(BaseLM):

    def __init__(self, device='cuda', pretrained='gpt2', revision='main', subfolder=None, tokenizer=None, batch_size=1):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(pretrained, revision=revision +("/" + subfolder if subfolder is not None else "")).to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hardcoding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained if tokenizer is None else tokenizer, revision=revision, subfolder=subfolder)

        assert isinstance(self.tokenizer, (
            transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
            transformers.T5Tokenizer, transformers.T5TokenizerFast,
        )), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size
        self.eot_token_id = self.tokenizer.eos_token_id # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        self.max_gen_toks = 256

        try:
            self.max_length = self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparantly
            self.max_length = self.gpt2.config.max_position_embeddings

        if isinstance(self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
            assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373], self.tokenizer.encode('hello\n\nhello')

        # multithreading and batching
        gpus = torch.cuda.device_count()
        batch_size_per_gpu = batch_size # todo: adaptive batch size

        # TODO: fix multi-gpu
        self.batch_size = batch_size_per_gpu# * gpus

        # TODO: fix multi-gpu
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)
    
    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits retuned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0][:, :, :50257]
    
    def _model_generate(self, context, max_length, eos_token_id):
        return self.gpt2.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )


# for backwards compability
GPT2LM = HFLM