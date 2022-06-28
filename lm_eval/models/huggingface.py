import math
import torch
import torch.nn.functional as F
import transformers
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from lm_eval.api import utils
from lm_eval.api.model import TokenLM, TokenSequence


class HuggingFaceAutoLM(TokenLM):

    AUTO_MODEL_CLASS: transformers.AutoModel = None

    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        device: Optional[str] = "cuda",
        half: Optional[bool] = True,
        batch_size: Optional[int] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        parallelize: Optional[bool] = False,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(half, bool)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        self.model = self.create_auto_model(pretrained, revision, subfolder)
        self.tokenizer = self.create_auto_tokenizer(
            pretrained, revision, subfolder, tokenizer
        )

        self._batch_size = batch_size  # TODO: adaptive batch size
        self._device = torch.device(device)
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self.tokenizer.model_max_length = self.max_length

        self.model.eval()
        torch.set_grad_enabled(False)

        # TODO: Fix multi-gpu support.
        if half:
            self.model.half()
        if parallelize:
            self.model.parallelize()
            self._device = torch.device("cuda:0")
        else:
            self.model.to(self._device)

    def create_auto_model(
        self, pretrained: str, revision: str, subfolder: str
    ) -> transformers.AutoModel:
        """Returns a pre-trained pytorch model from a pre-trained model configuration."""
        return self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )

    def create_auto_tokenizer(
        self,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            subfolder=subfolder,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        # Model config has no seq length attribute; return the tokenizer's max length.
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Fix multi-gpu
        return self._batch_size  # * gpus

    @property
    def device(self) -> torch.device:
        # TODO: Fix multi-gpu
        return self._device

    def tok_encode(self, strings: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(strings, add_special_tokens=False)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings, padding=True, add_special_tokens=False, return_tensors="pt"
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)
        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False), self.batch_size
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop_sequences = request_args["stop_sequences"]
            max_generation_length = request_args["max_generation_length"]
            num_fewshot = request_args["num_fewshot"]

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None
            assert isinstance(num_fewshot, int) or num_fewshot is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None or num_fewshot == 0:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            # Ensure that the context does not encroach into the `space`
            # for the generation.
            token_context = self.tok_encode_batch(context)
            input_ids = token_context["input_ids"][
                :, self.max_gen_toks - self.max_length :
            ].to(self.device)
            attention_mask = token_context["attention_mask"][
                :, self.max_gen_toks - self.max_length :
            ].to(self.device)

            responses = self._model_generate(
                inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)


class AutoCausalLM(HuggingFaceAutoLM):
    """Causal language modeling.
    You can find a set of supported models in the HF documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForCausalLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def create_auto_tokenizer(
        self,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = super().create_auto_tokenizer(
            pretrained, revision, subfolder, tokenizer
        )
        tokenizer.padding_side = "left"
        return tokenizer

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)["logits"]

    def _model_generate(
        self, inputs: TokenSequence, max_tokens: int, stop: Optional[List[str]] = None
    ) -> TokenSequence:
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop)
        generations = self.model.generate(
            **inputs,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        TODO: Currently only works for relative position encoded Seq2Seq models.
        """
        if self._max_length is not None:
            return self._max_length
        return self._DEFAULT_MAX_LENGTH

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)
            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -(self.max_length - 1) :]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][
                    :, -(self.max_length - 1) :
                ]
            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            contexts, conts = utils.split_and_pad_windows(
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also somehow figure out partial caching for
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(**inputs, labels=labels["input_ids"])

    def _model_generate(
        self, inputs: TokenSequence, max_tokens: int, stop: Optional[List[str]] = None
    ) -> Union[TokenSequence, List[str]]:
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop)
        generations = self.model.generate(
            **inputs,
            max_length=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(self, sequence: str, tokenizer: transformers.PreTrainedTokenizer):
        self.sequence = sequence
        self.sequence_id = tokenizer.encode(sequence)
        self.sequence_id_len = len(self.sequence_id) + 1
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_token_id = input_ids[0, -self.sequence_id_len :]
        last_tokens = self.tokenizer.decode(last_token_id)
        is_stopped = self.sequence in last_tokens
        return is_stopped


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer, stop_sequences: List[str]
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(sequence, tokenizer)
                for sequence in stop_sequences
            ],
        ]
    )
