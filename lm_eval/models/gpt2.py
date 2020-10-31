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
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(device=args.get("device", "cpu"))

    def generate(self, context, max_gen_length, truncate=True):
        # when too long to fit in context, truncate from the left
        context_tensor = torch.tensor([self.tokenizer.encode(context.strip())[max_gen_length - 1024:]], dtype=torch.long).to(self.device)
        res = self.gpt2.generate(
            context_tensor,
            # TODO: change to have until rather than using eos_token_id
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            max_length=self.num_tokens(context) + max_gen_length,
        )

        # chop off the prompt and the final eos token
        return self.tokenizer.decode(res[0][min(1024 - max_gen_length, len(context_tensor[0])):-1]).strip()

    def loglikelihood(self, context, continuation, truncate=True):
        # when too long to fit in context, truncate from the left
        inp = torch.tensor([self.tokenizer.encode(context + continuation)[-1024:]], dtype=torch.long).to(self.device)
        ctxlen = len(self.tokenizer.encode(context.strip()))

        cont_toks = inp[:, ctxlen:]  # [batch, seq]
        logits = F.log_softmax(self.gpt2(inp)[0], dim=-1)[:, ctxlen - 1:-1]  # [batch, seq, vocab]

        return torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)

    def num_tokens(self, string):
        return len(self.tokenizer.tokenize(string))
