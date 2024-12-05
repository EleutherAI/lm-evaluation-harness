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
        max_tokens: int = MAX_TOKENS_DEFAULT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_gen_tokens: int = DEFAULT_MAX_GEN_TOKENS,
        add_bos_token: bool = False,
        context_prefix_cache: bool = False,
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
        self.batch_size = int(batch_size)
        self.max_gen_tokens = max_gen_tokens
        self.add_bos_token = add_bos_token
        self.context_prefix_cache = context_prefix_cache
        self.backend = "causal"

    def _longest_common_prefix(self, list_of_strings):
        for a in range(1, len(list_of_strings[0])):
            try:
                if not all(
                    letter.startswith(list_of_strings[0][:a])
                    for letter in list_of_strings[1:]
                ):
                    return list_of_strings[0][: a - 1]
            except IndexError:
                return list_of_strings[0][: a - 1]

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    def _preserve_last_target_len_scores(
        self, scores, prompt_lengths: List, non_padding_lengths: List
    ) -> Tuple:
        """

        :param scores: logits (or log probs)
        :param prompt_lengths: a list of prompt token lengths for each input string that corresponded to an item in the
                               first dimension of the scores (the batch item)
        :param non_padding_lengths: a list of the lengths of the tokenizations of input and label concatenations
                                    for each such item
        :return: Return the scores where every item in the last 2 dimensions (i.e., for every item in the batch) is all
                 zero if it corresponds to anything other than the last n-logit vocabulary scores, the conditional
                 probability of producing a continuation of n tokens given the input, where n is the number of
                 continuation tokens for the batch item

        Intuitively, it masks out all but the last n target tokens in the given scores from later calculations

        also returns a mask for the score sequence positions (the 2nd dimension) that correspond to the tokens of the
        target
        """
        import mlx.core as mx

        batch_size, logits_seq_len, vocab_size = scores.shape
        assert all(
            [
                length - prompt_length <= logits_seq_len
                for prompt_length, length in zip(prompt_lengths, non_padding_lengths)
            ]
        )
        indices = mx.stack([mx.arange(logits_seq_len)] * batch_size)
        target_pos = list(
            map(
                lambda i: logits_seq_len - (i[0] - i[1]),
                zip(non_padding_lengths, prompt_lengths),
            )
        )
        target_mask = indices >= mx.array(target_pos)[..., None]
        zeros = mx.zeros_like(scores)
        expanded_mask = mx.repeat(target_mask[..., None], vocab_size, axis=2)
        result = mx.where(expanded_mask, scores, zeros)
        return result, target_mask

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
            from mlx_lm.models.cache import make_prompt_cache
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
        if self.context_prefix_cache:
            # Calculate (and store) a common prefix to use for the entire run
            common_prefix = self._longest_common_prefix(
                [context + continuation for (context, continuation), _, _ in requests]
            )
            cache = make_prompt_cache(self.model, 4096)
            processed = 0
            step_size = 512
            y = mx.array(self.tokenizer.encode(common_prefix))
            while y.size > 0:
                self.model(y[:step_size][None], cache=cache)
                mx.eval([c.state for c in cache])
                mx.metal.clear_cache()
                processed += min(y.size, step_size)
                y = y[step_size:]
            eval_logger.info(
                f"Cached common prefix of '{common_prefix}': "
                f"{len(self.tokenizer.encode(common_prefix, add_special_tokens=False)):,} tokens"
            )
            eval_logger.info(f"Peak memory: {mx.metal.get_peak_memory() / 1e9:.3f} GB")

        for chunk in chunks:
            full_sequences = []
            prompt_lengths = []
            cont_toks_list = []

            for _, context_enc, continuation_enc in chunk:
                full_sequence = context_enc + continuation_enc
                full_sequences.append(full_sequence)
                prompt_lengths.append(len(context_enc))
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

            target_only_log_probs, target_masks = self._preserve_last_target_len_scores(
                log_probs, prompt_lengths, lengths
            )
            # [current_batch_size, max_length_in_batch-1, vocab] and [current_batch_size, max_length_in_batch-1]

            _, full_seq_len, vocab_size = logits.shape
            flattened_shape = current_batch_size, full_seq_len
            cont_lengths = [len(i) for i in cont_toks_list]
            assert all(i == cont_lengths[0] for i in cont_lengths)
            target_seq_size = max(cont_lengths)
            arr = mx.take_along_axis(
                log_probs,
                mx.array(
                    [[0] * (log_probs.shape[1] - len(i)) + i for i in cont_toks_list]
                )[..., None],
                -1,
            )
            # Obtain log-probs at the corresponding continuation token indices
            # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
            # logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
            #     -1
            # )  # [1, seq]
            target_sequence_pos = (
                mx.stack([mx.arange(full_seq_len)] * current_batch_size)
                >= full_seq_len - target_seq_size
            )
            # print(log_probs.shape, target_sequence_pos.shape, flattened_shape, arr.shape)
            # print(target_sequence_pos)
            target_log_probs = mx.where(
                target_sequence_pos,
                arr.reshape(*flattened_shape),
                mx.zeros(flattened_shape),
            ).sum(1)[..., None]

            for (
                (
                    request_str,
                    ctx_tokens,
                    _,
                ),
                answer_target_log_probs,
                answer_greedy_tokens,
                cont_toks,
            ) in zip(chunk, target_log_probs, all_greed_tokens, cont_toks_list):
                # Check if per-token argmax for final scores associated with length of cont_toks
                # is exactly equal to cont_toks
                greedy_target_tokens = answer_greedy_tokens[-len(cont_toks) :]
                cont_toks = mx.array(cont_toks)
                max_equal = (greedy_target_tokens == cont_toks).all()

                # Answer: (log prob, is-exact-match)
                # Sum over conditional log likelihood values for final n tokens, where n is the length of cont_toks)
                answer_score = answer_target_log_probs.item()
                answer = (answer_score, bool(max_equal))
                res.append(answer)
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
        raise NotImplementedError("generate_until is not implemented")
