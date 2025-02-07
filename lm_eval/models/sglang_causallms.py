from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance

@register_model("sglang")
class SGLangLM(TemplateLM):
    def __init__(
        self,
        pretrained: str,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        max_model_len: int = None,
        batch_size: int = 1,
        **kwargs
    ):
        super().__init__()
        # Initialize your sglang model here
        self.model_name = pretrained
        self.batch_size = batch_size
        # ... other initialization code

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
        # Return the model's maximum sequence length
        return 2048  # Replace with the actual maximum length

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