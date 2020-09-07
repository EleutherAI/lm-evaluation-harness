import transformers
import torch
from ..base import LM
from . import MODEL_REGISTRY


@MODEL_REGISTRY.register("gpt2")
class GPT2LM(LM):
    def __init__(self):
        self.gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    
    def generate(self, context, max_gen_length):
        context = torch.tensor([self.tokenizer.encode(context.strip())], dtype=torch.long)
        res = self.gpt2.generate(
            context,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            max_length=max_gen_length,
        )

        # chop off the prompt and the final eos token
        return self.tok.decode(res[0][len(context[0]):-1]).strip()

    def loglikelihood(self, context, continuation):
        pass
