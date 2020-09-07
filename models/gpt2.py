import transformers
import torch
import torch.nn.functional as F
from ..base import LM
from .. import utils
from . import MODEL_REGISTRY


@MODEL_REGISTRY.register("gpt2")
class GPT2LM(LM):
    def __init__(self, device="cpu"):
        self.gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        self.device = device

    @classmethod
    def create_from_args(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(device=args.get("device", "cpu"))

    def generate(self, context, max_gen_length):
        context = torch.tensor([self.tok.encode(context.strip())], dtype=torch.long).to(self.device)
        res = self.gpt2.generate(
            context,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            max_length=max_gen_length,
        )

        # chop off the prompt and the final eos token
        return self.tok.decode(res[0][len(context[0]):-1]).strip()

    def loglikelihood(self, context, continuation):
        inp = torch.tensor([self.tok.encode(context + continuation)], dtype=torch.long).to(self.device)
        ctxlen = len(self.tok.encode(context.strip()))

        cont_toks = inp[:, ctxlen:] # [batch, seq]
        logits = F.log_softmax(self.gpt2(inp)[0], dim=-1)[:, ctxlen - 1:-1]  # [batch, seq, vocab]

        return torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
