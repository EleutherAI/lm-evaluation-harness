from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, eval_logger

from .huggingface import HFLM


eval_logger = eval_logger
TOP_P_DEFAULT = 1
MAX_TOKENS_DEFAULT = 2048
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_GEN_TOKENS = 256


@register_model("mlx", "mlx_lm")
class MLX(TemplateLM):
    tokenizer_name = HFLM.tokenizer_name
    apply_chat_template = HFLM.apply_chat_template
    tok_encode = HFLM.tok_encode

    def __init__(
        self,
        model: str = None,
        adapter_path: str = None,
        trust_remote_code: bool = False,
        eos_token: str = None,
        top_p: int = TOP_P_DEFAULT,
        max_tokens: int = MAX_TOKENS_DEFAULT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_gen_tokens: int = DEFAULT_MAX_GEN_TOKENS,
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
        self,
        logits,
        prompt_lengths: List,
        non_padding_lengths: List,
        dtype: np.dtype = np.float32,
    ) -> Tuple:
        """

        :param logits: logits
        :param prompt_lengths: a list of prompt token lengths for each input string that corresponded to an item in the
                               first dimension of the logits (the batch item)
        :param non_padding_lengths: a list of the lengths of the tokenizations of input and label concatenations
                                    for each such item
        :return: Return the logits where every item in the last 2 dimensions (i.e., for every item in the batch) is all
                 zero if it corresponds to anything other than the last n-logit vocabulary scores, the conditional
                 probability of producing a continuation of n tokens given the input, where n is the number of
                 continuation tokens for the batch item

        Intuitively, it masks out all but the last n target tokens in the given logits from later calculations

        also returns a mask for the logits sequence positions (the 2nd dimension) that correspond to the tokens of the
        target
        """
        import mlx.core as mx
        import numpy as np

        batch_size, logits_seq_len, vocab_size = logits.shape
        target_logits = np.zeros((logits.shape[0], logits_seq_len, vocab_size), dtype)
        target_mask = np.zeros((logits.shape[0], logits_seq_len, vocab_size), np.int32)
        for i in range(batch_size):
            prompt_length = prompt_lengths[i]
            non_padding_length = non_padding_lengths[i]
            target_length = non_padding_length - prompt_length
            for j in range(logits_seq_len):
                if (
                    j > (min(non_padding_length, logits_seq_len) - target_length)
                    or (logits_seq_len - j) <= target_length
                ):
                    target_logits[i][j] = logits[i][j]
                    target_mask[i][j] = 1
        return mx.array(target_logits), mx.array(target_mask)

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
            import mlx.nn as nn
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
            group_by=None,  # "contexts",
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
                cont_toks_list.append(continuation_enc)

            current_batch_size = len(full_sequences)
            lengths = [len(x) for x in full_sequences]

            # Pad to the largest
            max_length_in_batch = min(max(lengths), self.max_tokens)

            batch_arr = np.zeros((current_batch_size, max_length_in_batch), np.int32)

            # Left-padded
            for j in range(current_batch_size):
                truncated_length = min(lengths[j], self.max_tokens)
                batch_arr[j, -truncated_length:] = full_sequences[j][:truncated_length]
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

            # log softmax probabilities
            log_probs = nn.log_softmax(logits, axis=-1)

            all_greed_tokens = log_probs.argmax(axis=-1)
            # [current_batch_size, max_length_in_batch-1]

            target_only_log_probs, target_masks = self._preserve_last_target_len_logits(
                log_probs, prompt_lengths, lengths
            )
            # [current_batch_size, max_length_in_batch-1, vocab] and [current_batch_size, max_length_in_batch-1]

            for (
                (
                    request_str,
                    ctx_tokens,
                    _,
                ),
                answer_target_log_probs,
                inplen,
                cont_toks,
                answer_greedy_tokens,
                target_mask,
            ) in zip(
                chunk,
                target_only_log_probs,
                inplens,
                cont_toks_list,
                all_greed_tokens,
                target_masks,
            ):
                # Check if per-token argmax for final scores associated with length of cont_toks
                # is exactly equal to cont_toks
                greedy_target_tokens = answer_greedy_tokens[-len(cont_toks) :]
                cont_toks = mx.array(cont_toks)
                num_cont_toks = len(cont_toks)
                max_equal = (greedy_target_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                # logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                #     -1
                # )  # [1, seq]
                answer_target_log_probs = answer_target_log_probs[
                    mx.arange(num_cont_toks) - num_cont_toks, cont_toks
                ]

                # Answer: (log prob, is-exact-match)
                # Sum over conditional log likelihood values for final n tokens, where n is the length of cont_toks)
                answer_score = answer_target_log_probs.sum().item()
                answer = (answer_score, bool(max_equal))
                res.append(answer)
                pbar.update(1)

        pbar.close()
        return res  # re_ord.get_original(res)

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
