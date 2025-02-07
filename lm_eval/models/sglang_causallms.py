import copy
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from importlib.util import find_spec
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
)
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    handle_stop_sequences,
    undistribute,
)
from tqdm import tqdm

try:
    import sglang as sgl
    from sglang.srt.server_args import ServerArgs
    from sglang.lang.ir import SglSamplingParams
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass

@register_model("sglang")
class SGLangLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        # batch args from lm-eval interface: https://github.com/EleutherAI/lm-evaluation-harness/blob/144a1e58be73f937f8fecaae886346681d0fa082/docs/interface.md
        batch_size: Union[str, int] = 1,
        max_batch_size= None,
        max_model_len: int = None,
        max_gen_toks: int = 256,
        ########## SGlang native args ##########
        # Todo(Jinwei): Include more args of SGLang Engine if needed. Refer to https://docs.sglang.ai/backend/server_arguments.html .
        tokenizer_path: Optional[str] = None,
        tokenizer_mode: str = "auto",
        load_format: str = "auto",
        trust_remote_code: bool = True,
        dtype: str = "auto",
        kv_cache_dtype: str = "auto",
        context_length: Optional[int] = None,
        device: str = "cuda",
        # Memory and scheduling
        mem_fraction_static: Optional[float] = None,
        # parallelism
        dp_size: int = 1,
        tp_size: int = 1,
        **kwargs
    ):
        super().__init__()

        if not find_spec("sglang"):
            raise ModuleNotFoundError(
                "attempted to use 'sglang' LM type, but package `sglang` is not installed. "
                "Please install sglang via `pip install lm-eval[sglang]` or `pip install -e .[sglang]`"
            )
        
        assert "cuda" in device or device is None, "SGLang only supports CUDA"
        assert context_length is None or max_model_len is None, (
            "Either context_length or max_model_len may be provided, but not both"
        )
        # Initialize your sglang model here
        self._max_length = max_model_len if max_model_len is not None else context_length
        self.tensor_parallel_size = int(tp_size)
        self.data_parallel_size = int(dp_size)
        self.model_args = {
            "model_path": pretrained,
            "tokenizer_path": tokenizer_path,
            "tokenizer_mode": tokenizer_mode,
            "load_format": load_format,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "kv_cache_dtype": kv_cache_dtype,
            "context_length": int(self._max_length) if self._max_length else None,
            "device": device,
            "mem_fraction_static": mem_fraction_static,
            "tp_size": self.tensor_parallel_size,
            "dp_size": self.data_parallel_size,
        }

        self.model_args.update(kwargs)
        server_args = ServerArgs(**self.model_args)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        if self.data_parallel_size <= 1:
            self.model = sgl.Engine(**server_args)
        else:
            eval_logger.warning(
                "Data parallelism will be deprecated in the future version of SGLang. See here: https://docs.sglang.ai/backend/server_arguments.html#data-parallelism ."
            )
            raise NotImplementedError("Data parallelism is not supported for SGLang models now.")
        
        # Todo(Jinwei): check tokenizer and other settings.
        self.tokenizer = self.model.tokenizer_manager.tokenizer
        self._max_gen_toks = max_gen_toks


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Implement loglikelihood calculation
        # Return [(log_prob, is_greedy), ...]
        mocked = [(0.0, True)] * len(requests)
        return mocked

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        # Implement rolling loglikelihood calculation
        # Return [log_prob, ...]
        mocked = [0.0] * len(requests)
        return mocked

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        eos = self.tokenizer.decode(self.eot_token_id)
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            # perform batched generation
            cont = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )

            # cache generations
            for output, context in zip(cont, context):
                generated_text = output.outputs[0].text
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):  
        # check sglang sampling parammeters: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/sampling/sampling_params.py#L21  and https://docs.sglang.ai/references/sampling_params.html.
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            sampling_params = {
                "max_new_tokens": max_tokens, 
                "stop": stop,
            }
            sampling_params.update(kwargs)
        else:
            sampling_params = {
                "temperature": 0,
                "max_new_tokens": 1,
            }
            sampling_params.update(kwargs)
        if self.data_parallel_size > 1:
            raise NotImplementedError("Data parallelism is not supported for SGLang models now.")

        # Refer to: https://github.com/sgl-project/sglang/blob/0a6f18f068e4095fc228e798454e8496c9749214/python/sglang/srt/entrypoints/engine.py#L111 
        outputs = self.model.generate(
            input_ids=requests,
            sampling_params=sampling_params,
            
        )
        return outputs
    
    @property
    def eot_token_id(self):
        # Return the EOT (End of Text) token ID
        return None  # Replace with the actual EOT token ID

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.tokenizer_manager.context_length
        else:
            raise NotImplementedError("Data parallelism is not supported for SGLang models now.")
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        # Return the maximum number of tokens for generation
        return self._max_gen_toks

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding: Union[List[List[int]], List[int]] = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_decode(self, tokens: List[int]) -> str:
        # Implement token-to-text decoding
        pass

    @property
    def tokenizer_name(self) -> str:
        """
        Return the name of the model's tokenizer and/or the accompanying chat template.
        The returned string is used to cache requests.

        Returns:
            str: The name of the model's tokenizer and/or chat template.
        """
        pass
    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        """
        Get the appropriate chat template for the model based on the `chat_template` argument.

        This method returns the chat template string to build the prompt from a chat history.
        The chat template is saved in the evaluation results for reproducibility.
        Boolean arguments should be used with models that have only one chat template,
        while string arguments are used with models that have multiple chat templates.
        For the reference implementation, see HFLM class in `lm_eval.models.huggingface`.

        Args:
            chat_template (Union[bool, str]): Specifies whether to apply a chat template:
                - If False: Do not apply any chat template.
                - If True: Apply the default chat template.
                - If str: Apply the specified chat template by name.

        Returns:
            str: The selected chat template in Jinja format.
        """
        pass
    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Process a chat history to create a string that can be tokenized and input into the model.

        Args:
            chat_history (List[Dict[str, str]]): A list of dictionaries representing the chat history,
                where each dictionary has "role" and "content" keys.

        Returns:
            str: A string representing the chat history that can be tokenized and fed into the model.
        """
        pass