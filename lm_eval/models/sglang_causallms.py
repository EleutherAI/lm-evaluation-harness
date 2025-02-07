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


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Implement loglikelihood calculation
        # Return [(log_prob, is_greedy), ...]
        pass

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        # Implement rolling loglikelihood calculation
        # Return [log_prob, ...]
        pass

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # Implement text generation
        # Return a list of generated texts
        pass

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
        return 256  # Replace with the actual maximum generation length

    def tok_encode(self, string: str) -> List[int]:
        # Implement text-to-token encoding
        pass

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