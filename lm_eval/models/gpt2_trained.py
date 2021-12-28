import transformers
import torch
from lm_eval.base import BaseLM


class TrainedLM(BaseLM):

    def __init__(self, device='cuda', pretrained='gpt2', revision='main', subfolder=None, tokenizer=None, batch_size=1):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        self._device = torch.device(device)

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.gpt2 = transformers.GPT2LMHeadModel.from_pretrained("/home/muops/datasets/eleuther/models/1.5B").to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0][:, :, :50257]
    
    def _model_generate(self, context, max_length, eos_token_id):
        return (self.gpt2.generate(
            context,
            max_length=max_length-len(context),
            eos_token_id=eos_token_id,
            do_sample=False
        ))
