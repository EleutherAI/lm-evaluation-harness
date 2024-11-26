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
        model,
        adapter_path=None,
        trust_remote_code=False,
        eos_token=None,
        top_p=1,
        max_tokens=2048,
        batch_size=4,
        max_gen_tokens=256,
        add_bos_token=False,
        verbose=False,
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
            model.load_weights(adapter_path, strict=False)
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

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=None,
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc=f"Running loglikelihood requests ({len(requests):,} batches)",
        )
        chunks = re_ord.get_batched(n=self.batch_size)
        for chunk in chunks:
            full_sequences = []
            prompt_lengths = []

            for _, context_enc, continuation_enc in chunk:
                full_sequences.append(context_enc + continuation_enc)
                prompt_lengths.append(len(context_enc))

            current_batch_size = len(full_sequences)
            lengths = [len(x) for x in full_sequences]

            if max(lengths) > self.max_tokens:
                print(
                    f"[WARNING] Some sequences are longer than {self.max_tokens} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {self.max_tokens}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the largest
            max_length_in_batch = min(max(lengths), self.max_tokens)

            batch_arr = np.zeros((current_batch_size, max_length_in_batch), np.int32)

            for j in range(current_batch_size):
                truncated_length = min(lengths[j], self.max_tokens)
                batch_arr[j, :truncated_length] = full_sequences[j][:truncated_length]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )
            batch = mx.array(batch_arr)
            non_padding_lengths = mx.array(lengths)
            prompt_lengths = mx.array(prompt_lengths)

            shifted_padded_full_sequence = batch[
                :, :-1
            ]  # all but the last token for each sequence
            logits = self.model(shifted_padded_full_sequence)
            logits = logits.astype(mx.float32)

            # log softmax probabilities
            log_probs = mx.log(mx.softmax(logits, axis=-1))

            # Create mask to exclude padding and inputs
            mask_width = shifted_padded_full_sequence.shape[1]
            flattened_token_indices = mx.arange(mask_width)
            token_indices = flattened_token_indices[None, :]
            mask = mx.logical_and(
                token_indices >= prompt_lengths[:, None],
                token_indices < non_padding_lengths[:, None],
            )

            batch_greedy_tokens = logits.argmax(axis=-1)
            # A sequence of 1s or 0's the same width as the batch, where 1 indicates the target token is the same
            # as the greedily-generated token (determined efficiently via argmax on token probabilities)
            masked_indicator_values = (
                batch_greedy_tokens == shifted_padded_full_sequence
            ) * mask

            # A sequence of booleans indicating whether the sum of indicator values is equal to corresponding
            # target length
            batch_target_is_greedy_values = masked_indicator_values.sum(
                axis=-1
            ) == mx.array(non_padding_lengths)

            # Iterate over question, answer pairs, their log softmax logits, and their greedy values
            for idx, (is_greedy, log_prob) in enumerate(
                zip(batch_target_is_greedy_values, log_probs)
            ):
                prompt_length = prompt_lengths[idx].item()
                full_sequence_length = non_padding_lengths[idx].item()
                target_length = full_sequence_length - prompt_length

                # Extract target sequence and log prob scores that correspond to the predicted probability for it
                target_log_prob_scores = log_prob[-target_length:]
                target_sequence = full_sequences[idx][
                    prompt_length : full_sequence_length + 1
                ]

                # Use the target sequence for extracting log prob values from logits vocabulary distribution
                reshaped_target_seq = mx.reshape(mx.array(target_sequence), (-1, 1))
                target_log_probs = mx.take_along_axis(
                    target_log_prob_scores, reshaped_target_seq, axis=1
                )
                # Sum over conditional log likelihood values for target sequences
                answer_score = target_log_probs.squeeze(1).sum().item()

                # Answer: (log prob, is-exact-match)
                res.append((answer_score, is_greedy.item()))
            pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
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
