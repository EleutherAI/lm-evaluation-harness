import torch
from lm_eval.base import BaseLM
from tqdm import tqdm
import torch
import torch.nn.functional as F
from composer.core.precision import get_precision_context, Precision
from typing import Optional
from lm_eval import utils
import transformers
import inspect
class ComposerLLM(BaseLM):
    def __init__(
        self,
        model, # Can be any torch module whose forward expects a dict w/ keys ['input_ids', 'attention_mask']
        tokenizer, # Can be any tokenizer whose forward method returns a dict w/ keys ['input_ids', 'attention_mask']
        device,
        batch_size=4,
        precision: Optional[str] = None,
    ):
        super().__init__()

        self.precision = precision

        assert isinstance(device, str)
       

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size 


    @property
    def eot_token_id(self):
        res = self.tokenizer.pad_token_id
        if res is None:
            res = self.tokenizer.eos_token_id
        if res is None:
            return self.tokenizer.vocab_size - 1
        return res            
      
    @property
    def max_length(self):
        if hasattr(self.model, 'cfg'):
            return self.model.cfg.max_seq_len
        else:
            return self.model.config.max_position_embeddings
       
    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        truncation = True
        return [
                x for x in self.tokenizer(string,
                truncation=truncation,
            )['input_ids']
            if x != self.tokenizer.bos_token_id
        ]

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        forward_argspec = inspect.getfullargspec(self.model.forward).args
        args = {"input_ids": inps}
        if 'key_padding_mask' in forward_argspec:
            # composer gpt uses key padding mask
            args['key_padding_mask'] =  ~(inps == self.eot_token_id)
        elif 'attention_mask' in forward_argspec:
            # huggingface transformer uses attention_mask
            args['attention_mask'] =  ~(inps == self.eot_token_id)

        
        with torch.no_grad():
            if self.precision is not None:
                with get_precision_context(self.precision):
                    res = self.model(**args)
            else:
                res = self.model(**args)
                
            if isinstance(res, transformers.modeling_outputs.CausalLMOutputWithPast):
                res = res.logits
            return res[:, :, :self.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        return super()._loglikelihood_tokens(requests, padding_length=self.max_length, padding_token=self.eot_token_id)