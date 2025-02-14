import tensorrt_llm
import torch

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from tensorrt_llm.runtime import ModelRunnerCpp
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Tuple

@register_model('tensorrt_llm', 'trt_llm')
class TensorRTLLM(TemplateLM):
    def __init__(
        self,
        engine_dir: str = None,
        tokenizer: str = None,
        batch_size: int = 1,
        device: str = "cuda",
        # TODO: Add sampling params etc.
    ) -> None:
        super().__init__()
        # TODO: Allow specification of logger level
        tensorrt_llm.logger.set_level('info')
        assert engine_dir is not None, 'Please specify a path containing your TensorRT-LLM model\'s engine.\nFor more information on creating engines, please visit TensorRT-LLM\'s webpage:\nhttps://github.com/NVIDIA/TensorRT-LLM'
        assert tokenizer is not None, "Please specify a valid tokenizer for your model"
        self.runtime_rank = tensorrt_llm.mpi_rank()
        runner_kwargs = dict(
            engine_dir=engine_dir,
            rank=self.runtime_rank,
        )
        self.batch_size = batch_size
        self.device = device
        self.model_runner = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.end_token = self.tokenizer.decode(self.eot_token_id)
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.eot_token_id
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def max_gen_toks(self) -> int:
        return 256

    def tok_encode(
        self,
        string: str, 
        **kwargs
        # TODO: Add flag for special tokens
    ) -> List[int]:
        return self.tokenizer.encode(string)

    def _loglikelihood_tokens(
        self,
        requests,
        **kwargs
    ) -> List[Tuple[float, bool]]:

        res = []
        for _, context_enc, continuation_enc in tqdm(requests):

            # how this all works (illustrated on a causal decoder-only setup):
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # model  \               \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

            # when too long to fit in context, truncate from the left
            input_enc = (context_enc + continuation_enc)[-(self.tokenizer.model_max_length + 1) :][:-1]
            input_len = len(input_enc)
            continuation_len = len(continuation_enc)

            trt_inp = torch.tensor(
                [input_enc],
                dtype=torch.int32,
                device=self.device
            )
            output_dict = self.model_runner.generate(
                max_new_tokens=1,
                batch_input_ids=trt_inp,
                end_id=self.eot_token_id,
                pad_id=self.pad_id,
                streaming=False,
                return_dict=True,
                output_sequence_lengths=True
            )
            context_logits = torch.nn.functional.log_softmax(output_dict['context_logits'][0], dim=-1)
            context_logits = context_logits.squeeze()
            logits = context_logits[input_len - continuation_len : input_len]
            logits = logits.unsqueeze(0)
            greedy_tokens = logits.argmax(dim=-1)

            continuation_enc = torch.tensor(
                continuation_enc,
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            max_equal = (greedy_tokens == continuation_enc).all()
            logits = torch.gather(logits, 2, continuation_enc.unsqueeze(-1)).squeeze(-1)
            answer = (float(logits.sum()), bool(max_equal))

            if self.runtime_rank == 0:
                res.append(answer)

        return res

    def loglikelihood_rolling(
        self, 
        requests, 
        disable_tqdm: bool = False
    ) -> List[float]:
        pass

    def generate_until(
        self,
        requests,
        disable_tqdm: bool = False
    ) -> List[str]:

        res = []
        for req in tqdm(requests):
            context = req.args[0]
            gen_kwargs = req.args[1]
            until = gen_kwargs['until']
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected 'until' to be of type Union[str,list] but got {until}")
            temperature = gen_kwargs.get("temperature", 0.0)

            context_enc = self.tokenizer.encode(context, add_special_tokens=True)
            context_len =  len(context_enc)

            if 'max_gen_toks' in gen_kwargs.keys():
                max_new_tokens = gen_kwargs['max_gen_toks']
            else:
                max_new_tokens = self.max_gen_toks

            trt_inp = torch.tensor(
                [context_enc],
                dtype=torch.int32,
                device=self.device
            )

            stop_words_list = [until]
            stop_words_list = tensorrt_llm.runtime.decode_words_list(stop_words_list, self.tokenizer)

            output_dict = self.model_runner.generate(
                max_new_tokens=max_new_tokens,
                end_id=self.eot_token_id,
                pad_id=self.pad_id,
                batch_input_ids=trt_inp,
                streaming=False,
                return_dict=True,
                output_sequence_lengths=True,
                stop_words_list=stop_words_list,
            )

            output_ids = output_dict['output_ids']
            output_ids = output_ids.squeeze()
            output_begin = context_len
            gen_seq = self.tokenizer.decode(output_ids[output_begin:])

            for term in until:
                if len(term) > 0:
                    gen_seq = gen_seq.split(term)[0]

            res.append(gen_seq)

        return res