from functools import lru_cache
from typing import Dict, List, Tuple

import jinja2
import numpy as np
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import eval_logger


eval_logger = eval_logger


@register_model("mlx", "mlx_lm")
class MLX(TemplateLM):
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
        ensure_bos_token=False,
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
        self.model, self.tokenizer = load(
            model, adapter_path=adapter_path, tokenizer_config=tokenizer_config
        )
        eval_logger.info(f"Model type is '{type(self.model)}")
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.batch_size = int(batch_size)
        self.max_gen_tokens = max_gen_tokens
        self.ensure_bos_token = ensure_bos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @lru_cache(maxsize=2000)
    def tok_encode(self, string: str) -> List[int]:
        encoding = self.tokenizer.encode(string)
        if self.ensure_bos_token and encoding[0] != self.tokenizer.bos_token_id:
            encoding = [self.tokenizer.bos_token_id] + encoding
        return encoding

    @property
    def chat_template(self) -> str:
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.chat_template
        return self.tokenizer.default_chat_template

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )

        return chat_templated

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is implemented")

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
        try:
            import mlx.core as mx
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'mlx' LM type, but package `mlx` is not installed. Please install mlx "
                "via `pip install 'lm-eval[mlx]'` or `pip install -e '.[mlx]'`"
            )
        if not requests:
            return []
        res = []
        # Keep order for later
        original_order = {
            (context, continuation): idx
            for idx, (context, continuation) in enumerate(
                [req.args for req in requests]
            )
        }

        info = {req.args: None for req in requests}

        # sort the requests by their length (changes order)
        idx = sorted(
            range(len(requests)),
            key=lambda i: len(requests[i].args[0]) + len(requests[i].args[1]),
        )
        if len(requests) < self.batch_size:
            raise ValueError(
                f"Dataset must have at least batch_size={self.batch_size}"
                f" examples but only has {len(requests)}."
            )

        # Make the batches:
        batch_idx = [
            idx[i : i + self.batch_size]
            for i in range(0, len(idx) - self.batch_size + 1, self.batch_size)
        ]
        if len(idx) % self.batch_size != 0:
            # If the total requests size is not a multiple of the batch size, there will be a final
            # batch with the remainder
            batch_idx.append(idx[self.batch_size * len(batch_idx) :])
        eval_logger.info(
            f"{len(requests):,} requests and {len(batch_idx):,} {self.batch_size:,}-or-less-item batches"
        )

        # randomize the batches
        indices = np.random.permutation(len(batch_idx))

        pbar = tqdm(
            total=len(indices),
            disable=(disable_tqdm or (self.rank != 0)),
            desc=f"Running loglikelihood requests ({len(batch_idx):,} batches)",
        )
        for i in indices:
            context_batch = []
            continuation_batch = []

            for j in batch_idx[i]:
                context, continuation = requests[j].args
                context_batch.append(context)
                continuation_batch.append(continuation)
            (
                batch,
                input_lengths,
                target_lengths,
                non_padding_lengths,
            ) = self.delineated_batches(
                self.batch_size, context_batch, continuation_batch
            )

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
                token_indices >= input_lengths[:, None],
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
            ) == mx.array(target_lengths)

            # Iterate over question, answer pairs, their log softmax logits, and their greedy values
            for idx, (is_greedy, log_prob) in enumerate(
                zip(batch_target_is_greedy_values, log_probs)
            ):
                input_length = input_lengths[idx].item()
                target_length = target_lengths[idx]
                context = context_batch[idx]
                continuation = continuation_batch[idx]

                del info[(context, continuation)]

                # Extract log prob scores at token sequence positions in the logits
                target_end_idx = input_length + target_length
                target_sequence = self.tok_encode(continuation)
                target_log_prob_scores = log_prob[
                    input_length - 1 : target_end_idx - 1, :
                ]

                # Use the target sequence for extracting log prob values from logits vocabulary distribution
                reshaped_target_seq = mx.reshape(mx.array(target_sequence), (-1, 1))
                target_log_probs = mx.take_along_axis(
                    target_log_prob_scores, reshaped_target_seq, axis=1
                )
                # Sum over conditional log likelihood values for target sequences
                answer_score = target_log_probs.squeeze(1).sum().item()

                idx = original_order[(context, continuation)]

                # Answer: (original index, log prob, is-exact-match)
                answer = idx, answer_score, is_greedy.item()
                res.append(answer)
            pbar.update(1)

        pbar.close()
        eval_logger.info(f"{len(res):,} response objects")
        eval_logger.info(f"{len(info):,} unprocessed")
        # Return the answers in the original order (lost by the batch creation process, which )
        return list(map(lambda i: i[1:], sorted(res, key=lambda i: i[0])))

    def delineated_batches(self, batch_size, context_text, continuation_text):
        try:
            import mlx.core as mx
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'mlx' LM type, but package `mlx` is not installed. Please install mlx via "
                "`pip install 'lm-eval[mlx]'` or `pip install -e '.[mlx]'`",
            )

        batch_size = min(batch_size, len(context_text))
        encoded_context_batch = [self.tok_encode(record) for record in context_text]
        encoded_continuation_batch = [
            self.tok_encode(record) for record in continuation_text
        ]

        input_lengths = [len(x) for x in encoded_context_batch]
        target_lengths = [len(x) for x in encoded_continuation_batch]

        full_labels = [
            encoded_context_batch[idx] + encoded_continuation_batch[idx]
            for idx in range(batch_size)
        ]
        lengths = [len(x) for x in full_labels]

        if max(lengths) > self.max_tokens:
            print(
                f"[WARNING] Some sequences are longer than {self.max_tokens} tokens. "
                f"The longest sentence {max(lengths)} will be truncated to {self.max_tokens}. "
                "Consider pre-splitting your data to save memory."
            )
        pad_to = 8
        max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)

        batch_arr = np.zeros((batch_size, max_length_in_batch), np.int32)
        adjusted_lengths = []
        for j in range(batch_size):
            input_length = input_lengths[j]
            full_ids_end_idx = input_length + min(
                target_lengths[j], max_length_in_batch - input_length
            )
            adjusted_lengths.append(full_ids_end_idx)
            batch_arr[j, :full_ids_end_idx] = full_labels[j][:full_ids_end_idx]

        batch = mx.array(batch_arr)
        input_lengths = mx.array(input_lengths)
        non_padding_lengths = mx.array(adjusted_lengths)

        return batch, input_lengths, target_lengths, non_padding_lengths

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
            if "until" in request_args:
                raise NotImplementedError("Support for until not implemented!")
            temperature = request_args.get("temperature", 0.0)
            verbose = request_args.get("verbose", False)
            res.append(
                generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    temperature,
                    request_args.get(self.max_tokens),
                    verbose,
                    top_p=self.top_p,
                )
            )
        return res
