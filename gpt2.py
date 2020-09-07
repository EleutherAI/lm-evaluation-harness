import transformers
from base import LM
import torch
import torch.nn.functional as F


class GPT2LM(LM):
    def __init__(self):
        self.gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        self.tok = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    
    def generate(self, context, until):
        context = torch.tensor([self.tok.encode(context.strip())], dtype=torch.long)
        res = self.gpt2.generate(context, eos_token_id=self.tok.encoder[until], do_sample=False, max_length=1024)

        # chop off the prompt and the final eos token
        return self.tok.decode(res[0][len(context[0]):-1]).strip()

    def loglikelihood(self, context, continuation):
        print('likelihood:', context, continuation)
        inp = torch.tensor([self.tok.encode(context + continuation)], dtype=torch.long)
        ctxlen = len(self.tok.encode(context.strip()))

        cont_toks = inp[:, ctxlen:] # [batch, seq]
        logits = F.log_softmax(self.gpt2(inp)[0], dim=-1)[:, ctxlen - 1:-1] # [batch, seq, vocab]

        return torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
