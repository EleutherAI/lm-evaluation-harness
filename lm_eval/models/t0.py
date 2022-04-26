import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm
import numpy as np
import math

class T0LM(LM):
    MAX_GEN_TOKS = 256
    MAX_INP_LENGTH = 512
    VOCAB_SIZE = 32100
    EOT_TOKEN_ID = 1

    def __init__(self, device='cuda', parallelize=False, pretrained='t0', batch_size=1):
        super().__init__()
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(pretrained)
        self.t0 = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained)
        self.t0.eval()

        if parallelize == "True":
            print(parallelize)
            self.t0.parallelize()
            self.device = torch.device('cuda:0')
        else:
            self.t0.to(self.device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained)
        self.max_length = self.MAX_INP_LENGTH

        self.batch_size = int(batch_size)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def loglikelihood(self, requests):
        res = []
        for chunk in tqdm(utils.chunks(requests, self.batch_size), total=math.ceil(len(requests)/self.batch_size)):

            inputs, targets = zip(*chunk)

            inputs_tok = self.tokenizer(
                list(inputs),
                max_length=self.max_length,
                padding=True,
                # truncation=True,
                add_special_tokens=False,
                return_tensors="pt"
                ).to(self.device)

            for key in inputs_tok:
                inputs_tok[key] = inputs_tok[key][:, -(self.max_length - 1) :]

            targets_tok = self.tokenizer(
                list(targets),
                max_length=self.MAX_GEN_TOKS,
                padding=True,
                # truncation=True,
                add_special_tokens=False,
                return_tensors="pt"
                ).to(self.device)

            for key in targets_tok:
                targets_tok[key] = targets_tok[key][:, -(self.max_length - 1) :]

            with torch.no_grad():
                outputs = self.t0(
                    **inputs_tok,
                    labels=targets_tok["input_ids"]
                    )

            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)
            
            output_iterator = zip(
                chunk,
                log_softmaxes,
                targets_tok["input_ids"],
                targets_tok["attention_mask"],
            )
            for cache_key, log_softmax, target_tok, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tok = target_tok[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tok).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tok.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
            
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return res
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _get_stopping_criteria(self, stopping_criteria_ids):
        class MultitokenEOSCriteria(transformers.StoppingCriteria):
            def __init__(self, eos_seq_id: torch.LongTensor, tokenizer):
                self.eos_seq = tokenizer.decode(eos_seq_id)
                self.eos_seq_id = eos_seq_id
                self.eos_seq_len = len(eos_seq_id) + 1
                self.tokenizer = tokenizer

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                last_token_id = input_ids[0, -self.eos_seq_len:]
                last_tokens = self.tokenizer.decode(last_token_id)
                is_stopped = self.eos_seq in last_tokens
                return is_stopped
        
        class EOSCriteria(transformers.StoppingCriteria):
            def __init__(self, eos_token_id: torch.LongTensor):
                self.eos_token_id = eos_token_id

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                return input_ids[0,-1] == self.eos_token_id
         
        return transformers.StoppingCriteriaList([
            MultitokenEOSCriteria(stopping_criteria_ids, self.tokenizer),
            EOSCriteria(self.tokenizer.eos_token)
        ])

    def greedy_until(self, requests):
        
        res = []

        for context, until in tqdm(requests):
            if isinstance(until, str): until = [until]

            context_enc = self.tokenizer(context, return_tensors="pt").to(self.device).input_ids

            stopping_criteria_ids = self.tokenizer.encode(until[0])
            stopping_criteria = self._get_stopping_criteria(stopping_criteria_ids)

            cont = self.t0.generate(
                context_enc,
                max_length=self.MAX_GEN_TOKS,
                stopping_criteria=stopping_criteria,
                do_sample=False
            )

            s = self.tokenizer.decode(cont[0].tolist())
            
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            
            res.append(s)
        
        return res