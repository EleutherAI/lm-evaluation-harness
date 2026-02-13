import abc
import hashlib
import json
import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from tqdm import tqdm

from lm_eval import utils


if TYPE_CHECKING:
    from sqlitedict import SqliteDict

    from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)

T = TypeVar("T", bound="LM")


class LM(abc.ABC):
    """Abstract base class for language models.

    Subclasses take text (strings) as input and yield strings as output.
    Inputs and outputs should be tokenization-agnostic.
    """

    def __init__(self) -> None:
        # set rank and world size to a single process, by default.
        self._rank = 0
        self._world_size = 1
        self._device = None
        self.cache_hook: CacheHook = CacheHook(None)

    @abc.abstractmethod
    def loglikelihood(self, requests: list["Instance"]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.

        Downstream tasks should prefer this over other LM calls whenever possible.

        Args:
            requests: List of ``Instance`` objects. Each ``Instance.args`` is a ``(context, continuation)`` tuple.
                *context* — the conditioning text (implementations must handle empty string).
                *continuation* — the text to score. Word-boundary spaces belong in the
                continuation (e.g. ``context="hello"  continuation=" world"``).

        Returns:
            A list of ``(logprob, is_greedy)`` tuples — the log-probability of
            the continuation and whether it would be produced by greedy decoding.
        """
        pass

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests: list["Instance"]) -> list[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation.

        - Uses the full max context length of the model.
        - Inputs exceeding that length are chunked, up to the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.

        Example::

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

        Args:
            requests: List of ``Instance`` objects. Each ``Instance.args`` is a ``(string,)`` tuple containing
                the text whose overall log-likelihood is computed.

        Returns:
            A list of ``(logprob,)`` tuples — the log-probability of the string
            conditioned on the BOS/EOS token (or ``prefix_token_id``).
        """
        pass

    # TODO: Add an optional max length
    @abc.abstractmethod
    def generate_until(self, requests: list["Instance"]) -> list[str]:
        """Generate greedily until a stopping sequence.

        Args:
            requests: List of ``Instance`` objects. Each ``Instance.args`` is a ``(context, gen_kwargs)`` tuple.
                *context*: str — the conditioning text.
                *gen_kwargs*: str — generation keyword arguments (e.g. ``top_k``, ``until``).

        Returns:
            A list of generated continuation strings, one per request.
        """
        pass

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt=True
    ) -> str:
        """Transform few-shot chat history into a string prompt for the model.

        Args:
            chat_history: Messages as ``[{"role": ..., "content": ...}, ...]`` dicts.
            add_generation_prompt: Whether to append an assistant generation prefix
                (e.g. ``<|assistant|>``). Set to False when prefilling an assistant message.

        Returns:
            The formatted prompt string.
        """
        raise NotImplementedError(
            "To use this model with chat templates, please implement the 'apply_chat_template' method for your model type."
        )

    @classmethod
    def create_from_arg_string(
        cls: type[T], arg_string: str, additional_config: dict | None = None
    ) -> T:
        """Create an LM instance from a comma-separated argument string.

        Args:
            arg_string: Arguments as ``"key1=value1,key2=value2"``.
            additional_config: Extra configuration merged into the parsed args.

        Returns:
            An instance of this LM subclass.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    @classmethod
    def create_from_arg_obj(
        cls: type[T],
        arg_dict: dict[str, Any],
        additional_config: dict[str, Any] | None = None,
    ) -> T:
        """Create an LM instance from a dictionary of arguments.

        Args:
            arg_dict: Keyword arguments forwarded to the constructor.
            additional_config: Extra configuration merged into ``arg_dict``.

        Returns:
            An instance of this LM subclass.
        """
        additional_config = (
            {}
            if additional_config is None
            else {k: v for k, v in additional_config.items() if v is not None}
        )

        return cls(**arg_dict, **additional_config)

    # Distributed communication primitives, used by evaluate() to synchronize across ranks;
    # Override if relying on the evaluator() for data-parallel distribution of requests.
    # Otherwise, model methods receive all requests and can handle distribution internally.

    @property
    def device(self):
        return self._device

    @property
    def rank(self) -> int:
        """Index of this process. Default: 0 (single-process)."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Total number of processes. Default: 1 (single-process)."""
        return self._world_size

    def all_gather(self, tensor):
        """All-gather a tensor across ranks.

        Returns concatenated tensor from all ranks. Default: no-op.
        """
        return tensor

    def gather_object(self, obj, dst=0):
        """Gather a Python object to ``dst`` rank.

        Returns list of objects on ``dst``, None on others. Default: returns ``[obj]``.
        """
        return [obj]

    def barrier(self) -> None:
        """Synchronization barrier. Default: no-op."""
        return

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer or chat template, used to fingerprint request caches.

        Required for subclasses that support chat templating.
        """
        raise NotImplementedError(
            "To use this model with chat templates, please implement the 'tokenizer_name' property."
        )

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        """Return the chat template string for this model.

        Override in subclasses to define a specific format. Returns empty string
        by default (no chat template).
        """
        return ""

    def set_cache_hook(self, cache_hook: "CacheHook") -> None:
        self.cache_hook = cache_hook


### SQLite-based caching of LM responses
def hash_args(attr: str, args: Iterable[Any]) -> str:
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


class CacheHook:
    def __init__(self, cachinglm: Optional["CachingLM"]) -> None:
        if cachinglm is None:
            self.dbdict: SqliteDict | None = None
            return

        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr: str, req: Iterable[Any], res: Any) -> None:
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


class CachingLM:
    def __init__(self, lm: LM, cache_db: str) -> None:
        """LM wrapper that returns cached results when available, falling back to the underlying model.

        Args:
            lm: The underlying language model to wrap.
            cache_db: Path to the SQLite cache database.
        """
        from sqlitedict import SqliteDict

        self.lm: LM = lm
        self.cache_db: str = cache_db
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.dbdict = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr: str) -> Any:
        lm_attr = getattr(self.lm, attr)
        if attr not in ["loglikelihood", "loglikelihood_rolling", "generate_until"]:
            eval_logger.debug(f"Passing through attribute '{attr}' to underlying LM")
            return lm_attr

        def _fn(requests: list["Instance"]) -> list["Instance"]:
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
            rem_res = getattr(self.lm, attr)(remaining_reqs) if remaining_reqs else []

            # stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res, strict=True):
                while res[resptr] is not None:
                    resptr += 1

                res[resptr] = r

                # caching
                hsh = hash_args(attr, req.args)
                self.dbdict[hsh] = r
            self.dbdict.commit()

            return res

        return _fn

    def get_cache_hook(self) -> "CacheHook":
        return CacheHook(self)


class TemplateLM(LM):
    """LM subclass that provides shared tokenization and scoring boilerplate.

    Handles context/continuation encoding, empty-context logic, and
    delegates token-level scoring to ``_loglikelihood_tokens``.
    """

    tokenizer = None
    backend = "causal"

    @property
    @abc.abstractmethod
    def eot_token_id(self) -> int:
        pass

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        return self.eot_token_id

    @abc.abstractmethod
    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        """Tokenize a string and return a list of token IDs.

        Must handle strings that already contain the BOS token when
        ``add_special_tokens`` is None. Otherwise, uses the flag as given.
        """
        pass

    @abc.abstractmethod
    def _loglikelihood_tokens(
        self, requests: list[tuple[tuple[str, str], list[int], list[int]]], **kwargs
    ) -> list[tuple[float, bool]]:
        pass

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """Encode a context-continuation pair into separate token ID lists.

        Trailing spaces in the context are moved to the continuation to preserve
        word-boundary tokenization. Causal models encode the full sequence then
        split; seq2seq models encode each part independently.

        Note:
            Does NOT handle empty context — the caller is responsible for that
            (see ``loglikelihood``).

        Args:
            context: The conditioning text (must be non-empty).
            continuation: The text to score.

        Returns:
            A ``(context_enc, continuation_enc)`` tuple of token ID lists.
        """
        assert context, "Context cannot be empty!"

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        if self.backend == "causal":
            whole_enc = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)

            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]
        else:
            # for SEQ2SEQ case we need to encode separately
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)

        return context_enc, continuation_enc

    def loglikelihood(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts.

        Tokenizes each ``(context, continuation)`` pair and delegates to
        ``_loglikelihood_tokens``. Empty contexts use ``prefix_token_id``
        (typically BOS/EOS) as the conditioning token.

        Args:
            requests: List of ``Instance`` objects. Each ``Instance.args`` is a ``(context, continuation)`` tuple.
            disable_tqdm: Whether to suppress the progress bar.

        Returns:
            A list of ``(logprob, is_greedy)`` tuples, one per request.
        """
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                continuation_enc = self.tok_encode(
                    continuation, add_special_tokens=False
                )
                # BOS or EOS as context: handle when context is empty -> (context + continuation) -> (BOS + continuation
                context_enc, continuation_enc = (
                    ([self.prefix_token_id], continuation_enc)
                    if self.prefix_token_id != continuation_enc[0]
                    else (continuation_enc[:1], continuation_enc[1:])
                )
                # BOS or EOS as context
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    @abc.abstractmethod
    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> list[float]:
        pass

    @abc.abstractmethod
    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        pass

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        """Select and return the appropriate chat template for this model.

        Resolution order (adapted from Transformers ``apply_chat_template``):

        * No tokenizer — returns the empty string (template handled by provider).
        * Tokenizer has a dict of templates — use the named or ``"default"`` entry.
        * Tokenizer has a single template — use it, falling back to
          ``default_chat_template`` if unset.

        Args:
            chat_template: ``False``/``None`` to disable, ``True`` to auto-select,
                or a string name to pick a specific template from a dict.

        Returns:
            The selected template string, or ``None`` if disabled.
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
        try:
            template = (
                self.tokenizer.chat_template or self.tokenizer.default_chat_template
            )
        except AttributeError:
            return None

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
