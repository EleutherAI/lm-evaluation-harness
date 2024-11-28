from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, eval_logger

from .huggingface import HFLM


eval_logger = eval_logger


@register_model("mlx", "mlx_lm")
class MLX(TemplateLM):
    tokenizer_name = HFLM.tokenizer_name
    apply_chat_template = HFLM.apply_chat_template
    tok_encode = HFLM.tok_encode

    def __init__(
        self,
        model: str = None,
        pretrained: str = None,
        adapter_path: str = None,
        trust_remote_code: bool = False,
        eos_token: str = None,
        top_p: int = 1,
        max_tokens: int = 2048,
        batch_size: int = 4,
        max_gen_tokens: int = 256,
        add_bos_token: bool = False,
        verbose: bool = False,
    ):
        try:
            from mlx_lm.utils import load
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'mlx' LM type, but package `mlx_lm` is not installed. please install mlx_lm via "
                "`pip install 'lm-eval[mlx]'` or `pip install -e '.[mlx]'`"
            )
        super().__init__()
        tokenizer_config = {"trust_remote_code": trust_remote_code}
        if eos_token is not None:
            tokenizer_config["eos_token"] = eos_token
        model = model or pretrained
        self.model, self.tokenizer = load(model, tokenizer_config=tokenizer_config)
        eval_logger.info(f"Model type is '{type(self.model)}")
        if adapter_path is not None:
            eval_logger.info(f"Loading pretrained adapters from {adapter_path}")
            self.model.load_weights(adapter_path, strict=False)
        self.max_tokens = int(max_tokens)
        self.top_p = top_p
        self.batch_size = int(batch_size)
        self.max_gen_tokens = max_gen_tokens
        self.add_bos_token = add_bos_token
        self.verbose = verbose
        self.backend = "causal"

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    def _preserve_last_target_len_logits(
        self, logits, prompt_lengths: List, non_padding_lengths: List
    ):
        """

        :param logits: logits
        :param prompt_lengths: a list of prompt token lengths for each input string that corresponded to an item in the
                               first dimension of the logits (the batch item)
        :param non_padding_lengths: a list of the lengths of the tokenizations of input and label concatenations
                                    for each such item
        :return: Return the logits where every item in the last 2 dimensions (i.e., for every item in the batch) is all
                 zero if it corresponds to anything other than the last n-logit vocabulary scores, the conditional
                 probability of producing a continuation of n tokens given the input

        Intuitively, it masks out all but the last n target tokens in the given logits from later calculations
        """
        import mlx.core as mx
        import numpy as np

        batch_size, logits_seq_len, vocab_size = logits.shape
        target_logits = np.zeros(
            (logits.shape[0], logits_seq_len, vocab_size), np.int32
        )
        for i in range(batch_size):
            prompt_length = prompt_lengths[i]
            non_padding_length = non_padding_lengths[i]
            target_length = non_padding_length - prompt_length
            for j in range(logits_seq_len):
                if (
                    (min(non_padding_length, logits_seq_len) - j) < target_length
                    if non_padding_length < logits_seq_len
                    else (min(non_padding_length, logits_seq_len) - j) <= target_length
                ):
                    target_logits[i][j] = logits[i][j]
        return mx.array(target_logits)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
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
        try:
            import mlx.core as mx
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'mlx' LM type, but package `mlx` is not installed. Please install mlx "
                "via `pip install 'lm-eval[mlx]'` or `pip install -e '.[mlx]'`"
            )

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
            group_by="contexts",
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
            desc=f"Running mlx loglikelihood requests ({len(requests):,})",
        )
        for chunk in chunks:
            full_sequences = []
            prompt_lengths = []
            inplens = []
            cont_toks_list = []

            for _, context_enc, continuation_enc in chunk:
                full_sequence = context_enc + continuation_enc
                full_sequences.append(full_sequence)
                prompt_lengths.append(len(context_enc))
                inplen = len(full_sequence[-(self.max_tokens + 1) :][:-1])
                inplens.append(inplen)
                cont_toks_list.append(context_enc)

            current_batch_size = len(full_sequences)
            lengths = [len(x) for x in full_sequences]

            # Pad to the largest
            max_length_in_batch = min(max(lengths), self.max_tokens)

            batch_arr = np.zeros((current_batch_size, max_length_in_batch), np.int32)

            for j in range(current_batch_size):
                truncated_length = min(lengths[j], self.max_tokens)
                batch_arr[j, :truncated_length] = full_sequences[j][:truncated_length]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )
            batch = mx.array(
                batch_arr
            )  # [current_batch_size, max_length_in_batch, vocab]

            shifted_padded_full_sequence = batch[
                :, :-1
            ]  # all but the last token for each sequence
            logits = self.model(shifted_padded_full_sequence).astype(mx.float32)
            # [current_batch_size, max_length_in_batch-1, vocab]

            target_only_logits = self._preserve_last_target_len_logits(
                logits, prompt_lengths, lengths
            )
            # [current_batch_size, max_length_in_batch-1, vocab]

            # log softmax probabilities
            log_probs = mx.log(mx.softmax(target_only_logits, axis=-1))

            for (
                request_str,
                ctx_tokens,
                _,
            ), answer_target_log_probs, inplen, cont_toks in zip(
                chunk, log_probs, inplens, cont_toks_list
            ):
                # Check if per-token argmax for final scores associated with length of cont_toks
                # is exactly equal to cont_toks
                greedy_tokens = answer_target_log_probs.argmax(axis=-1)[
                    -len(cont_toks) :
                ]

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=answer_target_log_probs,
                ):
                    assert isinstance(ctx_tokens, list)
                    max_equal = (greedy_tokens == mx.array(cont_toks)).all()

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))
                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood_rolling is not implemented")

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """
        * Each request contains Instance.args : Tuple[str, dict] containing 1. an input string to the LM and 2. a
          dictionary of keyword arguments used to control generation parameters.
        * Using this input and these generation parameters, text will be sampled from the language model
          (typically until a maximum output length or specific stopping string sequences--for example,
          {"until": ["\n\n", "."], "max_gen_toks": 128}).
        * The generated input+output text from the model will then be returned.
        """
        try:
            from mlx_lm.utils import generate
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'mlx' LM type, but package `mlx` is not installed. Please install anthropic via "
                "`pip install 'lm-eval[mlx]'` or `pip install -e '.[mlx]'`",
            )

        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            prompt, request_args = request
            temperature = request_args.pop("temperature", 0.0)
            request_args.pop("do_sample", None)
            request_args.pop("until", None)
            res.append(
                generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.max_gen_tokens,
                    verbose=self.verbose,
                    temp=temperature,
                    **request_args,
                )
            )
        return res
