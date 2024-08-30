import abc
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

import transformers
from sqlitedict import SqliteDict
from tqdm import tqdm

from lm_eval import utils


eval_logger = logging.getLogger("lm-eval")

T = TypeVar("T", bound="LM")


class LM(abc.ABC):
    def __init__(self) -> None:
        """Defines the interface that should be implemented by all LM subclasses.
        LMs are assumed to take text (strings) as input and yield strings as output
        (inputs/outputs should be tokenization-agnostic.)

        """
        # set rank and world size to a single process, by default.
        self._rank = 0
        self._world_size = 1
        self.cache_hook = CacheHook(None)

    @abc.abstractmethod
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        pass

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests) -> List[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: BOS/EOS
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  BOS   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the BOS/EOS token.
                Can also be overridden for custom cases by `prefix_token_id`.
        """
        pass

    # TODO: Add an optional max length
    @abc.abstractmethod
    def generate_until(self, requests) -> List[str]:
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, gen_kwargs).
            context: str
                Context string
            gen_kwargs: dict
                A dictionary of keyword arguments to pass to the generation function e.g. top_k, until, etc.
        :return: list[str]
            A list of model generated continuations.
            continuation: str
                The generated continuation.
        """
        pass

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Defines how to transform few-shot examples provided as chat history into a format that can be used as input to the LM.

        :param chat_history: list[dict[str, str]]
            A list of dictionaries with keys 'role' and 'content'.
            Values are strings representing the role name and the content of the message, respectively.
        :return: str
            A string representing the chat history in a format that can be used as input to the LM.
        """
        raise NotImplementedError(
            "To use this model with chat templates, please implement the 'apply_chat_template' method for your model type."
        )

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    @classmethod
    def create_from_arg_obj(
        cls: Type[T], arg_dict: dict, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given arg_obj

        Parameters:
        - arg_obj: A dict containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """

        additional_config = {} if additional_config is None else additional_config
        additional_config = {
            k: v for k, v in additional_config.items() if v is not None
        }

        return cls(**arg_dict, **additional_config)

    @property
    def rank(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._rank

    @property
    def world_size(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        """Must be defined for LM subclasses which implement Chat Templating.
        Should return the name of the tokenizer or chat template used.
        Used only to properly fingerprint caches when requests are being cached with `--cache_requests`, otherwise not used.
        """
        raise NotImplementedError(
            "To use this model with chat templates, please implement the 'tokenizer_name' property."
        )

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        """Returns the chat template structure for user/assistant messages if a template is provided.
        This method is intended to be overridden in a subclass to define a specific chat template format.
        For models that do not support chat templates, this method returns None by default.
        """

        return ""

    def set_cache_hook(self, cache_hook) -> None:
        self.cache_hook = cache_hook


### SQLite-based caching of LM responses
def hash_args(attr, args):
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


class CacheHook:
    def __init__(self, cachinglm) -> None:
        if cachinglm is None:
            self.dbdict = None
            return

        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr, req, res) -> None:
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


class CachingLM:
    def __init__(self, lm, cache_db) -> None:
        """LM wrapper that returns cached results if they exist, and uses the underlying LM if not.

        :param lm: LM
            Underlying LM
        :param cache_db: str
            Path to cache db
        """
        self.lm = lm
        self.cache_db = cache_db
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.dbdict = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr: str):
        lm_attr = getattr(self.lm, attr)
        if attr not in ["loglikelihood", "loglikelihood_rolling", "generate_until"]:
            eval_logger.debug(f"Passing through attribute '{attr}' to underlying LM")
            return lm_attr

        def fn(requests):
            res = []
            remaining_reqs = []
            warned = False
            # figure out which ones are cached and which ones are new
            eval_logger.info(
                f"Loading '{attr}' responses from cache '{self.cache_db}' where possible..."
            )
            for req in tqdm(requests, desc="Checking cached requests"):
                hsh = hash_args(attr, req.args)
                if attr == "generate_until" and req.args[1].get("do_sample", False):
                    # when we are doing non-greedy generation, don't use the cache
                    # (else every "randomly sampled" generation would be identical for repeats > 1).
                    if not warned:
                        eval_logger.warning(
                            f"Arguments to lm.generate_until() '{req.args[1]}' include non-deterministic sampling. Caching will not be performed for such requests."
                        )
                        warned = True
                    res.append(None)
                    remaining_reqs.append(req)
                elif hsh in self.dbdict:
                    ob = self.dbdict[hsh]

                    assert ob is not None

                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append(req)
            eval_logger.info(
                f"Cached requests: {len(requests) - len(remaining_reqs)}, Requests remaining: {len(remaining_reqs)}"
            )
            # actually run the LM on the requests that do not have cached results
            rem_res = getattr(self.lm, attr)(remaining_reqs)

            # stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res):
                while res[resptr] is not None:
                    resptr += 1

                res[resptr] = r

                # caching
                hsh = hash_args(attr, req.args)
                self.dbdict[hsh] = r
            self.dbdict.commit()

            return res

        return fn

    def get_cache_hook(self):
        return CacheHook(self)


class TemplateLM(LM):
    """
    A class acting as intermediary between the LM base class
    and boilerplate often included in other LM subclasses.
    """

    tokenizer = None

    @property
    @abc.abstractmethod
    def eot_token_id(self):
        pass

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        return self.eot_token_id

    @abc.abstractmethod
    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """
        Tokenize a string using the model's tokenizer and return a list of token IDs.
        """
        pass

    @abc.abstractmethod
    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        pass

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        model_class = getattr(self, "AUTO_MODEL_CLASS", None)

        if model_class == transformers.AutoModelForSeq2SeqLM:
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        else:
            whole_enc = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)

            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    @abc.abstractmethod
    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> List[float]:
        pass

    @abc.abstractmethod
    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        pass

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        """
        Set and get the appropriate chat template for the model.
        This method sets the tokenizer's chat_template and returns the template string for reproducibility.

        The template selection logic is adapted from the Transformers library's `apply_chat_template`
        method in the Tokenizer class. The original implementation can be found at:
        https://github.com/huggingface/transformers/blob/fc35907f95459d7a6c5281dfadd680b6f7b620e3/src/transformers/tokenization_utils_base.py#L1687

        This method ensures that the right template is chosen based on the following:
        1. If the model's tokenizer has multiple templates:
            a. Use the specified template if it exists in the dictionary.
            b. Use the default template from the list if no specific template is provided.
            c. Raise an error if no default template exists and no specific template is provided.
        2. If the model's tokenizer has a single template or no template:
            a. Use the tokenizer's chat template if available.
            b. Fall back to the default chat template if no tokenizer chat template exists.

        Args:
            chat_template (Union[bool, str]): Specifies the chat template to use.
                - If False or None, no template is applied.
                - If True, the default or only available template is used.
                - If a string, the template with the matching name is used.

        Returns:
            Optional[str]: The selected chat template, or None if no template is applied.
        """
        if self.tokenizer is None:
            return ""

        if chat_template is False or chat_template is None:
            eval_logger.warning(
                "model.chat_template was called with the chat_template set to False or None. "
                "Therefore no chat template will be applied. Make sure this is an intended behavior."
            )
            return None

        # Convert boolean chat_template to None to ensure compatibility with the adapted logic
        if isinstance(chat_template, bool):
            chat_template = None
        using_default_template = False

        # First, handle the cases when the model has a dict of multiple templates
        template = self.tokenizer.chat_template or self.tokenizer.default_chat_template

        if isinstance(template, dict):
            using_default_dict = self.tokenizer.chat_template is None

            if chat_template is not None:
                if chat_template in template:
                    selected_template = template[chat_template]
                    if using_default_dict:
                        using_default_template = True
                else:
                    raise ValueError(
                        f"The specified chat template '{chat_template}' is not available. "
                        f"Available template names are {sorted(template.keys())}."
                    )
            else:
                # If user didn't pass a chat template, use the default template from the dict
                if "default" in template:
                    selected_template = template["default"]
                    using_default_template = True
                else:
                    raise ValueError(
                        "This model has multiple chat templates with no default specified! Please either pass a chat "
                        "template or the name of the template you wish to use to the `chat_template` argument. Available "
                        f"template names are {sorted(template.keys())}."
                    )

        # Cases when the model has a single template or no template
        else:
            # priority: `chat_template` argument > `tokenizer.chat_template` > `tokenizer.default_chat_template
            if isinstance(chat_template, str):
                eval_logger.warning(
                    "Chat template name provided, but the tokenizer's chat template is not a dictionary. "
                    "Using the tokenizer's chat template or the default template instead."
                )
            if self.tokenizer.chat_template is not None:
                selected_template = self.tokenizer.chat_template
            else:
                selected_template = self.tokenizer.default_chat_template
                using_default_template = True

        if using_default_template:
            eval_logger.warning(
                "No chat template is set for this tokenizer, falling back to a default class-level template. This is "
                "very error-prone, because models are often trained with templates different from the class default! "
                "Default chat templates are a legacy feature and will be removed in Transformers v4.43, at which "
                "point any code depending on them will stop working. We recommend setting a valid chat template before "
                "then to ensure that this model continues working without issues."
            )

        return selected_template
