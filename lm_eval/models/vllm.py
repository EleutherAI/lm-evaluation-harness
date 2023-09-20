from lm_eval import utils
from lm_eval.base import BaseLM
from typing import List, Mapping, NewType, Optional, Tuple, Union
from vllm import LLM, SamplingParams
import torch
from transformers import BatchEncoding
import transformers
from tqdm import tqdm

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

class VLLM(BaseLM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        add_special_tokens: Optional[bool] = None,
        batch_size: Optional[int] = 1,
        max_gen_toks: Optional[int] = 1024,
        max_length: Optional[int] = None,
        trust_remote_code: Optional[bool] = False,
        tensor_parallel_size: Optional[int] = 1,
        dtype: Optional[Union[str, torch.dtype]] = 'bfloat16',
    ):
        super().__init__()
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._batch_size = batch_size
        self._trust_remote_code = trust_remote_code
        self._add_special_tokens = add_special_tokens
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=self._trust_remote_code,
        )
        self.llm = LLM(model=pretrained, 
                       tensor_parallel_size=tensor_parallel_size,
                       dtype=dtype)
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=pretrained,
        )
    
    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        context = [r[0] for r in requests]
        until = requests[0][1]
        max_tokens = self.max_gen_toks
        token_context = self.tok_encode_batch(context)
        generated_texts = self._model_generate(
            inputs=token_context,
            max_tokens=max_tokens,
            stop=until,
            temperature=0.0,
        )
        for text in generated_texts:
            self.cache_hook.add_partial("greedy_until", (context, until), text)

        return generated_texts

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        num_return_sequences: int = 1,
        num_return_sequences_batch: int = -1, # Doesn't do anything. Just here to match the signature of the other models.
        temperature: float = 0.0, 
        top_p: float = 1,
    ) -> TokenSequence:

        if isinstance(stop, str):
            stop = [stop]

        input_ids = inputs["input_ids"][:, self.max_gen_toks-self.max_length:]

        # Decode each back to a string
        contexts = self.tok_decode(input_ids)

        bsz = len(input_ids)

        output_texts = []
        sampling_params = SamplingParams(max_tokens=max_tokens, 
                                         temperature=temperature, 
                                         top_p=top_p,
                                         stop=stop, 
                                         n=num_return_sequences)
        outputs = self.llm.generate(prompts=contexts, 
                                    sampling_params=sampling_params,
                                    use_tqdm=bsz > 1)
        
        # Sort by request_id
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        for output in outputs:
            generations = [gen.text for gen in output.outputs]
            if len(generations) == 1:
                generations = generations[0]
            output_texts.append(generations)

        return output_texts

    def generate(self, requests):
        # Two cases: either all requests are the same until, is_greedy, and _model_generate_kwargs
        # or they are all different. In the former case, we can batch the requests together
        # and call _model_generate once. In the latter case, we have to call _model_generate
        # for each request individually.

        contexts, untils, is_greedys, _model_generate_kwargss = zip(*[self.parse_request(request) for request in requests])
        # Hash the str of everything
        str_untils = [str(x) for x in untils]
        str_model_generate_kwargss = [str(x) for x in _model_generate_kwargss]
        if len(set(str_untils)) == 1 and len(set(is_greedys)) == 1 and len(set(str_model_generate_kwargss)) == 1:
            print("All generation parameters are the same. Batching.")
            # All requests are the same, so we can batch them together.
            context_enc = self.tok_encode_batch(contexts)
            generated_texts = self._model_generate(
                inputs=context_enc,
                max_tokens=self.max_gen_toks,
                stop=untils[0],
                **_model_generate_kwargss[0]
            )
            for request, generated_text in zip(requests, generated_texts):
                self.cache_hook.add_partial("generate", request, generated_text)
            return generated_texts
        else:
            print("Some generation parameters are different. Not batching.")
            # Requests are different, so we have to call _model_generate individually.
            results = []
            for request in tqdm(requests):
                context, until, is_greedy, _model_generate_kwargs = self.parse_request(request)

                context_enc = self.tok_encode_batch(context)
                generated_texts = self._model_generate(
                    inputs=context_enc,
                    max_tokens=self.max_gen_toks,
                    stop=until,
                    **_model_generate_kwargs
                )
                cache = (context, until, tuple(_model_generate_kwargs))
                self.cache_hook.add_partial("generate", cache, generated_texts)
                results.append(generated_texts)
            return results
        
    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=self._trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer
        
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return 'cuda' # I don't think this is used anywhere

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )

    def _model_call(self, inps):
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
