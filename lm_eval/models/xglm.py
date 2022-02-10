import transformers
import torch
from lm_eval import utils
from lm_eval.base import BaseLM
from tqdm import tqdm


class XGLM(BaseLM):
    def __init__(self, device='cuda', pretrained='facebook/xglm-1.7B', revision='main', subfolder=None, tokenizer=None, batch_size=1):
        super().__init__()
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)
        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.xglm = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            # cache_dir="/users/zyong2/data/zyong2/huggingface/xglm"
        ).to(self.device)
        print(f"🤖 Loading model {pretrained}")
        self.xglm.eval()
        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer, revision=revision, subfolder=subfolder)

        # assert isinstance(self.tokenizer, (
        #     transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
        #     transformers.T5Tokenizer, transformers.T5TokenizerFast,
        # )), "this tokenizer has not been checked for compatibility yet!"
        self.vocab_size = self.tokenizer.vocab_size
        # if isinstance(self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
        #     assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373], \
        #         self.tokenizer.encode('hello\n\nhello')
        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size
        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)
    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id
    @property
    def max_length(self):
        try:
            return self.xglm.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.xglm.config.max_position_embeddings
    @property
    def max_gen_toks(self):
        return 256
    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus
    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device
    def tok_encode(self, string: str):
        # HACK: to overcome problem of  XGLM tokenizer removing new lines
        # we replace newline with SEP token
        # WARNING: Since typical SEP token == EOS token
        # Generation will stop after the first appearance of SEP token prevnting XGLM from 
        # outputting Multi line generations
        string = string.replace("\n", self.tokenizer.sep_token)
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        # HACK: to overcome problem of  XGLM tokenizer removing new lines
        # replace back the generated sep_tokens with newlines
        output =  self.tokenizer.decode(tokens)
        output = output.replace(self.tokenizer.sep_token, "\n")
        print(output)
        return output
    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.xglm(inps)[0][:, :, :256008]
    
    def _model_generate(self, context, max_length, eos_token_id):
        result = self.xglm.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )
        return result
