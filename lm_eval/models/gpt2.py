import transformers
import torch
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils


class GPT2LM(LM):
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.gpt2.eval()
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(device=args.get("device", "cpu"))

    def loglikelihood(self, context, continuation, truncate=True):
        # when too long to fit in context, truncate from the left
        context_enc = self.tokenizer.encode(context)
        continuation_enc = self.tokenizer.encode(continuation)
        inp = torch.tensor([(context_enc + continuation_enc)[-1024:]], dtype=torch.long).to(self.device)
        ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - 1024)

        cont_toks = inp[:, ctxlen:]  # [batch, seq]
        logits = F.log_softmax(self.gpt2(inp)[0], dim=-1)[:, ctxlen - 1:-1]  # [batch, seq, vocab]

        return torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
