from lm_eval.api.model import LM, TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from typing import List, Tuple
from mlx_lm.utils import generate, load
from mlx_lm.generate import colorprint_by_t0
from mlx_lm.models.base import KVCache
from mlx_tuning_fork.tuning.utils import create_delineated_batches
import mlx.nn as nn
import mlx.core as mx

from tqdm import tqdm

eval_logger = utils.eval_logger


@register_model("mlx", "mlx_lm")
class MLX(TemplateLM):
    def __init__(self, model,  prompt_formatter, adapter_path=None, trust_remote_code=False, eos_token=None, top_p=1,
                 max_tokens=2048):
        super().__init__()
        tokenizer_config = {"trust_remote_code": trust_remote_code}
        if eos_token is not None:
            tokenizer_config["eos_token"] = eos_token
        self.prompt_formatter = prompt_formatter
        self.model, self.tokenizer = load(model, adapter_path=adapter_path, tokenizer_config=tokenizer_config)
        eval_logger.info(
            f"Model type is '{type(self.model)}"
        )

        self.max_tokens = max_tokens
        self.top_p = top_p

    def loglikelihood(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """
        * Each request contains Instance.args : Tuple[str, str] containing 1. an input string to the LM and 2. a target
          string on which the loglikelihood of the LM producing this target, conditioned on the input, will be returned.
        * Each request will have, as result, (ll, is_greedy): Tuple[float, int] returned, where ll is a floating point
          number representing the log probability of generating the target string conditioned on the input, and
          is_greedy being either the value 0 or 1, with it being 1 if and only if the target string would be generated
          by greedy sampling from the LM (that is, if the target string is the most likely N-token string to be output
          by the LM given the input. )
        """
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            new_reqs.append(((context, continuation), None, None))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
            self,
            requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
            disable_tqdm: bool = False,
            override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]


        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )

        kv_heads = (
            [self.model.n_kv_heads] * len(self.model.layers)
            if isinstance(self.model.n_kv_heads, int)
            else self.model.n_kv_heads
        )
        cache = [KVCache(self.model.head_dim, n) for n in kv_heads]

        def _step(y):
            logits = self.model(y[None], cache=cache)
            logits = logits[:, -1, :]
            token = mx.argmax(logits, axis=-1)
            return token

        for chunk in chunks:
            input_text = []
            output_text = []
            for context, continuation, _, _ in chunk:
                input_text.append(self.prompt_formatter.get_input(context))
                output_text.append(self.prompt_formatter.get_output(continuation))
            inputs, input_lengths, lengths = create_delineated_batches(input_text, output_text, self.tokenizer,
                                                                      max_seq_length=self.max_tokens)
            # Forward the concatenated [input, tagets] through the model.
            # Get just the logits for the targets by slicing the last targets.size columns
            # Compute use nn.losses.cross_entropy(sliced_logits, targets)
            shifted_inputs = inputs[:, :-1]
            shifted_labels = inputs[:, 1:]
            logits = self.model(shifted_inputs)
            logits = logits.astype(mx.float32)

            mask_width = shifted_inputs.shape[1]
            token_indices = mx.arange(mask_width)[None, :]
            mask = mx.logical_and(token_indices >= input_lengths[:, None], token_indices < lengths[:, None])

            ce = nn.losses.cross_entropy(logits, shifted_labels, reduction="sum") * mask

            for idx, loglikelihood in enumerate(ce):
                target_enc = self.tokenizer.encode(output_text[idx])
                input_enc = self.tokenizer.encode(input_text[idx])

                greedy_tokens = []
                y = _step(input_enc)
                mx.async_eval(y)
                greedy_tokens.append(y)
                while True:
                    next_y = _step(y)
                    if next_y != self.tokenizer.eos_token_id:
                        mx.async_eval(next_y)
                        greedy_tokens.append(next_y)
                        y = next_y
                    else:
                        break

                max_equal = target_enc == greedy_tokens

                # Answer: (log prob, is-exact-match)
                answer = (loglikelihood, max_equal)

                res.append(answer)
                pbar.update(1)

        pbar.close()

        return res

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("loglikelihood_rolling is not implemented")

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        * Each request contains Instance.args : Tuple[str, dict] containing 1. an input string to the LM and 2. a
          dictionary of keyword arguments used to control generation parameters.
        * Using this input and these generation parameters, text will be sampled from the language model
          (typically until a maximum output length or specific stopping string sequences--for example,
          {"until": ["\n\n", "."], "max_gen_toks": 128}).
        * The generated input+output text from the model will then be returned.
        """
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            prompt, request_args = request
            if "until" in request_args:
                raise NotImplementedError("Support for until not implemented!")
            temperature = request_args.get("temperature", 0.0)
            verbose = request_args.get("verbose", False)
            formatter = colorprint_by_t0 if request_args.get("colorize", False) else None
            res.append(
                generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    temperature,
                    request_args.get(self.max_tokens),
                    verbose,
                    formatter=formatter,
                    top_p=self.top_p))
        return res
