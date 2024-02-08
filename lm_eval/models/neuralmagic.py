from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from deepsparse import Pipeline
from deepsparse.utils.data import numpy_log_softmax


@register_model("sparseml")
class SparseMLLM(HFLM):
    """
    SparseML is an open-source model optimization toolkit that enables you to create
    inference-optimized sparse models using pruning, quantization, and distillation
    algorithms. Models optimized with SparseML can then be exported to the ONNX and
    deployed with DeepSparse for GPU-class performance on CPU hardware.
    """

    def _create_model(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        try:
            import sparseml
        except ModuleNotFoundError:
            raise Exception(
                "package `sparseml` is not installed. "
                "Please install it via `pip install sparseml[transformers]`"
            )

        # Load model with SparseAutoModel
        from sparseml.transformers.utils import SparseAutoModel
        from transformers import AutoConfig

        model_kwargs = kwargs if kwargs else {}
        ignored_kwargs = [
            "dtype",
            "parallelize",
            "device_map_option",
            "max_memory_per_gpu",
            "max_cpu_memory",
            "peft",
            "autogptq",
        ]
        for k in ignored_kwargs:
            model_kwargs.pop(k)

        config = AutoConfig.from_pretrained(pretrained)
        model = SparseAutoModel.text_generation_from_pretrained(
            pretrained, config=config, **model_kwargs
        )

        # Apply recipe to model
        # Note: Really annoying we can't grab the recipe.yaml present in the uploaded model
        # and you need this separate apply_recipe_structure_to_model function
        from sparseml.pytorch.model_load.helpers import apply_recipe_structure_to_model
        from huggingface_hub import hf_hub_download
        import os

        recipe_path = hf_hub_download(repo_id=pretrained, filename="recipe.yaml")
        apply_recipe_structure_to_model(
            model=model,
            recipe_path=recipe_path,
            model_path=os.path.dirname(recipe_path),
        )

        self._model = model

class DeepSparseLM(LM):
    def __init__(
        self,
        pipeline: Pipeline,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        tokenizer: Optional["AutoTokenizer"] = None,  # noqa: F821
    ):
        """
        Wrapper around the DeepSparse pipeline to make it compatible with the
        llm-evaluation-harness.

        :param pipeline: the pipeline object to wrap
        :param batch_size: the batch size to use for evaluation
        :param max_gen_toks: the maximum number of tokens to generate
            when using the model for generation (see: greed_until method)
        :param tokenizer: the tokenizer to use for encoding and decoding
            strings and tokens. By default, the tokenizer from the pipeline
        """
        super().__init__()

        self.pipeline = pipeline
        self.batch_size = batch_size
        self.tokenizer = tokenizer or pipeline.tokenizer
        self._max_length = pipeline.sequence_length
        self._max_gen_toks = max_gen_toks
        self.batch_sizes = {}

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

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
            list(utils.chunks(re_ord.get_reordered(), self.batch_size)),
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

            response = self.pipeline(
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

    def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "The method not required by any of our " "current task integrations so far"
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
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

            self._max_gen_toks = request_args.pop("max_gen_toks", self.max_gen_toks)

            for context, _ in chunk:
                # add context (prompts) to the list
                inps.append(context)

            until = request_args.pop("until", ["<|endoftext|>"])
            request_args.pop("do_sample", None)
            request_args["temperature"] = request_args.get("temperature", 0)

            # run inference (generate max_gen_toks tokens)
            out = self.pipeline(
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