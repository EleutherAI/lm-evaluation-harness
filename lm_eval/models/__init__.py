import logging
from typing import List, Mapping, Optional, Type

import lm_eval.api.utils
from lm_eval.api.model import LM

from . import dummy
from . import openai_completions
from . import huggingface


logger = logging.getLogger(__name__)


MODEL_API_REGISTRY = {
    "hf-causal": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "openai": openai_completions.OpenAICompletionsLM,
    "dummy": dummy.DummyLM,
}


def list_model_apis() -> List[str]:
    """Returns a list of all the model API names available for language model construction."""
    return sorted(list(MODEL_API_REGISTRY))


def get_model(model_api_name: str, **model_kwargs) -> LM:
    """Returns a language model from the specified model API, instantiated
    with the specified kwargs.

    Args
        model_api_name: Name of the model API to use as found in the model registry.
        **model_kwargs: Keyword arguments to pass to the model constructor. See constructor
            args for the model API in `lm_eval.models`.

    Returns:
        A language model instance.
    """
    model_api_class = _get_model_api_from_registry(model_api_name)
    return model_api_class(**model_kwargs)


def get_model_from_args_string(
    model_api_name: str,
    model_args: str,
    additional_config: Optional[Mapping[str, str]] = None,
) -> LM:
    """Returns a language model from the specified model API, instantiated with
    the given kwargs.

    Args:
        model_api_name: Name of the model API to use as found in the model registry.
        model_args: A string of comma-separated key=value pairs that will be passed
            to the model constructor. E.g. "pretrained=gpt2,batch_size=32".
        additional_config: An additional dictionary of key=value pairs that will be
            passed to the model constructor.

    Returns:
        A language model instance.
    """
    additional_config = {} if additional_config is None else additional_config
    additional_args = {k: v for k, v in additional_config.items() if v is not None}
    kwargs = lm_eval.api.utils.parse_cli_args_string(model_args)
    kwargs.update(additional_args)
    return get_model(model_api_name, **kwargs)


def _get_model_api_from_registry(model_api_name: str) -> Type[LM]:
    try:
        return MODEL_API_REGISTRY[model_api_name]
    except KeyError:
        logger.warning(f"Available model APIs:\n{list_model_apis()}")
        raise KeyError(f"Model API `{model_api_name}` is missing.")
