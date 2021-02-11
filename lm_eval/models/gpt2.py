import transformers
import torch
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm


class GPT2LM(LM):
    MAX_GEN_TOKS = 256

    def __init__(self, device="cpu", pretrained='gpt2'):
        self.device = torch.device(device)
        self.gpt2 = transformers.GPT2LMHeadModel.from_pretrained(pretrained).to(self.device)
        self.gpt2.eval()
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(pretrained)
        self.tokenizer.pad_token = "<|endoftext|>"

    @classmethod
    def create_from_arg_string(cls, arg_string):
        args = utils.simple_parse_args_string(arg_string)
        return cls(device=args.get("device", "cpu"), pretrained=args.get("pretrained", "gpt2"))

    def loglikelihood(self, requests):
        res = []
        with torch.no_grad():
            # TODO: vectorize properly
            # TODO: automatic batch size detection for vectorization
            for context, continuation in tqdm(requests):
                # when too long to fit in context, truncate from the left

                if context == "":
                    # end of text as context
                    context_enc = [50256]
                else:
                    context_enc = self.tokenizer.encode(context)
                
                continuation_enc = self.tokenizer.encode(continuation)
                inp = torch.tensor([(context_enc + continuation_enc)[-1024:]], dtype=torch.long).to(self.device)
                ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - 1024)

                cont_toks = inp[:, ctxlen:]  # [batch, seq]
                logits = F.log_softmax(self.gpt2(inp)[0], dim=-1)[:, ctxlen - 1:-1]  # [batch, seq, vocab]
                
                greedy_tokens = logits.argmax(dim=-1)
                max_equal = (greedy_tokens == cont_toks).all()

                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1) # [batch, seq]


                res.append((float(logits.sum()), bool(max_equal)))

        return res
    
    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles untils that are 
        # multiple tokens or that span multiple tokens correctly
        res = []

        for context, until in tqdm(requests):
            if isinstance(until, str): until = [until]

            context_enc = torch.tensor([self.tokenizer.encode(context)]).to(self.device)

            primary_until, = self.tokenizer.encode(until[0])

            cont = self.gpt2.generate(
                context_enc,
                max_length=self.MAX_GEN_TOKS,
                eos_token_id=primary_until,
                do_sample=False
            )

            s = self.tokenizer.decode(cont[0].tolist()[context_enc.shape[1]:])

            for term in until:
                s = s.split(term)[0]
            
            res.append(s)
        
        return res
