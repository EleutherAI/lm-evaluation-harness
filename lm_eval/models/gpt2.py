import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_eval.base import LM, TokenizedLM
from lm_eval import utils
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterable


class TorchLM(TokenizedLM):
    @abstractmethod
    def _model_generate(self, context, max_length, eos_token_id):
        pass

    @abstractmethod
    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits retuned from the model
        """
        pass

    # subclass must implement properties batch_size, vocab_size, eot_token_id, max_gen_toks, device.
    # TODO: enforce this somehow

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
            #   this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return (-len(toks), tuple(toks))
        
        # TODO: automatic (variable) batch size detection for vectorization
        reord = utils.Reorderer(requests, _collate)
        for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
            inps = []
            contlens = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9 <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the [:, -len(continuation_enc):, :self.vocab_size] slice
                # cont_toks      4 5 6 7 8 9

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length+1):][:-1]
                , dtype=torch.long).to(self.device)
                inplen, = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = padding_length if padding_length is not None else inplen

                # pad to length
                inp = torch.cat([
                    inp, # [seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device) # [padding_length - seq]
                ], dim=0)

                inps.append(inp.unsqueeze(0))
                contlens.append(cont)
                inplens.append(inplen)

            multi_logits = F.log_softmax(self._model_call(torch.cat(inps, dim=0)), dim=-1).cpu()  # [batch, seq, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                contlen = len(cont_toks)

                logits = logits[inplen-contlen:inplen].unsqueeze(0) # [1, seq, vocab]

                greedy_tokens = logits.argmax(dim=-1)

                # cont_toks :: [1, seq]
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)

                max_equal = (greedy_tokens == cont_toks).all()

                #last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1) # [1, seq]

                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return reord.get_original(res)
    
    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles untils that are 
        # multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return (len(toks), x[0])
        
        reord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(reord.get_reordered()):
            if isinstance(until, str): until = [until]

            primary_until, = self.tok_encode(until[0])
            
            context_enc = torch.tensor([self.tok_encode(context)[self.max_gen_toks - self.max_length:]]).to(self.device)

            cont = self._model_generate(context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until)

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1]:])

            for term in until:
                s = s.split(term)[0]
            
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            
            res.append(s)
        
        return reord.get_original(res)


class HFLM(TorchLM):

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