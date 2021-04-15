import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm


class GPT2LM(LM):
    MAX_GEN_TOKS = 256

    def __init__(self, device=None, pretrained='gpt2'):
        super().__init__()
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(pretrained).to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hardcoding this to gpt2
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = "<|endoftext|>"
        try:
            self.max_length = self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparantly
            self.max_length = self.gpt2.config.max_position_embeddings

        assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]

        # multithreading and batching
        gpus = torch.cuda.device_count()
        batch_size_per_gpu = 2 # todo: adaptive batch size

        self.batch_size = batch_size_per_gpu * gpus

        if gpus > 1:
            self.gpt2 = nn.DataParallel(self.gpt2)

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(device=args.get("device", None), pretrained=args.get("pretrained", "gpt2"))

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [50256]
            else:
                context_enc = self.tokenizer.encode(context)

            continuation_enc = self.tokenizer.encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(self, requests):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []
        with torch.no_grad():

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
            for chunk in utils.chunks(tqdm(reord.get_reordered()), self.batch_size):
                inps = []
                inplens = []
                ctxlens = []

                padding_length = None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor((context_enc + continuation_enc)[-self.max_length:], dtype=torch.long).to(self.device)
                    ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - self.max_length)
                    inplen, = inp.shape

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp, # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device) # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    inplens.append(inplen)
                    ctxlens.append(ctxlen)

                multi_logits = F.log_softmax(self.gpt2(torch.cat(inps, dim=0))[0][:, :, :50257], dim=-1)  # [batch, seq, vocab]

                for (cache_key, _, _), logits, ctxlen, inp, inplen in zip(chunk, multi_logits, ctxlens, inps, inplens):
                    logits = logits[ctxlen - 1:inplen - 1].unsqueeze(0) # [1, seq, vocab]

                    greedy_tokens = logits.argmax(dim=-1)
                    cont_toks = inp[:, ctxlen:inplen]  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    last_token_slice = logits[:, -1, :].squeeze(0).tolist()

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
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])
        
        reord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(reord.get_reordered()):
            if isinstance(until, str): until = [until]

            context_enc = torch.tensor([self.tokenizer.encode(context)[self.MAX_GEN_TOKS - self.max_length:]]).to(self.device)

            primary_until, = self.tokenizer.encode(until[0])

            cont = self.gpt2.generate(
                context_enc,
                max_length=context_enc.shape[1] + self.MAX_GEN_TOKS,
                eos_token_id=primary_until,
                do_sample=False
            )

            s = self.tokenizer.decode(cont[0].tolist()[context_enc.shape[1]:])

            for term in until:
                s = s.split(term)[0]
            
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            
            res.append(s)
        
        return reord.get_original(res)
