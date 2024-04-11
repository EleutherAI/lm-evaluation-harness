import copy
from typing import List, Optional, Tuple, Union

import numpy
import transformers
from tqdm import tqdm

import lm_eval.models.utils
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = utils.eval_logger


@register_model("sparseml")
class SparseMLLM(HFLM):
    """
    SparseML is an open-source model optimization toolkit that enables you to create
    inference-optimized sparse models using pruning, quantization, and distillation
    algorithms. Models optimized with SparseML can then be exported to the ONNX format and
    deployed with DeepSparse for GPU-class performance on CPU hardware.

    This class is a wrapper around the HuggingFace LM class to enable SparseML
    integration with the lm-evaluation-harness.
    """

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[str] = "auto",
        trust_remote_code: Optional[bool] = False,
        **kwargs,
    ) -> None:
        try:
            from sparseml.transformers import SparseAutoModelForCausalLM
        except ModuleNotFoundError:
            raise Exception(
                "Package `sparseml` is not installed. "
                "Please install it via `pip install sparseml[transformers]`"
            )

        model_kwargs = kwargs if kwargs else {}

        if "device_map" not in model_kwargs:
            # set a device_map to initialize model on the right GPU.
            # this is needed because it seems that the default behavior
            # for quantized models now seems to be device_map="auto"
            # which breaks data-parallel mode.
            if hasattr(self, "accelerator"):
                model_kwargs.update(
                    {"device_map": {"": f"cuda:{self.accelerator.local_process_index}"}}
                )
            else:
                model_kwargs.update({"device_map": {"": str(self.device)}})

        relevant_kwarg_names = [
            "offload_folder",
            "device_map",
        ]
        relevant_kwargs = {
            k: v for k, v in model_kwargs.items() if k in relevant_kwarg_names
        }

        # Log the difference between model_kwargs and relevant_kwargs so we can see
        # what is being ignored
        ignored_kwargs = {}
        for k, v in model_kwargs.items():
            if k not in relevant_kwargs.keys():
                ignored_kwargs[k] = v
        eval_logger.warning(
            f"The sparseml integration is ignoring the following kwargs that are specified: {ignored_kwargs}"
        )

        model = SparseAutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=lm_eval.models.utils.get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **relevant_kwargs,
        )
        self._model = model

    def _get_config(self, pretrained: str, **kwargs) -> None:
        try:
            from sparseml.transformers import SparseAutoConfig
        except ModuleNotFoundError:
            raise Exception(
                "Package `sparseml` is not installed. "
                "Please install it via `pip install sparseml[transformers]`"
            )

        self._config = SparseAutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained, **kwargs
        )

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        **kwargs,
    ) -> None:
        try:
            from sparseml.transformers import SparseAutoTokenizer
        except ModuleNotFoundError:
            raise Exception(
                "Package `sparseml` is not installed. "
                "Please install it via `pip install sparseml[transformers]`"
            )

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = SparseAutoTokenizer.from_pretrained(
                    tokenizer,
                    **kwargs,
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = SparseAutoTokenizer.from_pretrained(
                model_name,
                **kwargs,
            )
        return None


@register_model("deepsparse")
class DeepSparseLM(LM):
    """
    Wrapper around DeepSparse, a sparsity-aware deep learning
    inference runtime for CPUs, to make it compatible with the
    lm-evaluation-harness.
    """

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        batch_size: Optional[Union[int, str]] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
    ):
        super().__init__()

        try:
            import deepsparse
        except ModuleNotFoundError:
            raise Exception(
                "Package `deepsparse` is not installed. "
                "Please install it via `pip install deepsparse[transformers]`"
            )

        if isinstance(batch_size, str) and not batch_size.isdigit():
            eval_logger.warning(
                f"batch_size={batch_size} is not valid for deepsparse because it is not an integer. "
                "Ignoring and using the default of 1."
            )
            batch_size = 1

        self.batch_size = int(batch_size)
        self._max_length = max_length if max_length else self._DEFAULT_MAX_LENGTH
        self._max_gen_toks = max_gen_toks
        self.batch_sizes = {}

        # Initialize new model and tokenizer instances
        self.model = deepsparse.TextGeneration(
            model_path=pretrained,
            sequence_length=self._max_length,
            batch_size=batch_size,
        )
        self.tokenizer = tokenizer if tokenizer else self.model.tokenizer
        self.config = self.model.config

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Copied directly from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
        """
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                raise NotImplementedError(
                    "Implementing empty context is not supported yet"
                )
            context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        The function to compute the loglikelihood of the continuation
        tokens given the context tokens.

        This function is an adapted version of the original function from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
        """
        res = []

        def _collate(x):
            """Defines the key for the sorted method"""
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(lm_eval.models.utils.chunks(re_ord.get_reordered(), self.batch_size)),
            disable=disable_tqdm,
        ):
            batch_inp = []
            batch_cache_key = []
            batch_continuation_enc = []
            # len(chunk) is the batch_size
            for cache_key, context_enc, continuation_enc in chunk:
                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice # noqa: E501

                inp = (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]

                batch_inp.append(self.tokenizer.decode(inp))
                batch_cache_key.append(cache_key)
                batch_continuation_enc.append(continuation_enc)

            response = self.model(
                prompt=batch_inp,
                max_new_tokens=0,
                output_scores=True,
                include_prompt_logits=True,
            )

            for resp, continuation_enc, cache_key in zip(
                response.generations, batch_continuation_enc, batch_cache_key
            ):
                # (seq_len, vocab_size)
                multi_scores = resp.score

                from deepsparse.utils.data import numpy_log_softmax

                # (seq_len, vocab_size) but with softmax applied
                multi_logits = numpy_log_softmax(multi_scores, axis=1)
                # toss out the context half of the sequence
                # (cont_len, vocab_size)
                continuation_multi_logits = multi_logits[-len(continuation_enc) :]

                # pick out the logits for the continuation tokens
                # (cont_len,)
                continuation_logits = continuation_multi_logits[
                    numpy.arange(len(continuation_enc)), continuation_enc
                ]
                # check if the tokens generated greedly are the same
                # as the expected continuation
                greedy_tokens = continuation_multi_logits.argmax(axis=1)
                max_equal = greedy_tokens.tolist() == continuation_enc

                # Answer: (log prob, is-exact-match)
                answer = (float(continuation_logits.sum()), bool(max_equal))

                res.append(answer)

                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "The method not required by any of our current task integrations so far"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        The function to generate a certain number of new tokens
        given a context.

        This function is an adapted version of the original function from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py
        """
        if not requests:
            return []
        res = []
        requests = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        pbar = tqdm(total=len(requests))
        for chunk, request_args in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.batch_size))
        ):
            inps = []

            # make a deepcopy since we are changing arguments
            request_args = copy.deepcopy(request_args)

            self._max_gen_toks = request_args.pop("max_gen_toks", self.max_gen_toks)

            for context, _ in chunk:
                # add context (prompts) to the list
                inps.append(context)

            until = request_args.pop("until", ["<|endoftext|>"])
            request_args.pop("do_sample", None)
            request_args["temperature"] = request_args.get("temperature", 0)

            # run inference (generate max_gen_toks tokens)
            out = self.model(
                sequences=inps,
                max_new_tokens=self.max_gen_toks - 1,
                stop=until,
                **request_args,
            )

            for resp, (context, args_) in zip(out.generations, chunk):
                text = resp.text
                until_ = until
                # split the text at the first occurrence of any of the until tokens
                for term in until_:
                    if len(term) > 0:
                        text = text.split(term)[0]

                res.append(text)

                self.cache_hook.add_partial(
                    "generate_until", (context, {"until": until_}), text
                )
                pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        """
        Copied directly from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc
