import math
import transformers
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.base import BaseLM
from lm_eval import utils


class HuggingFaceAutoLM(BaseLM):

    AUTO_MODEL_CLASS: transformers.AutoModel = None

    def __init__(
        self,
        pretrained: str,
        tokenizer: transformers.PreTrainedTokenizer = None,
        subfolder: str = None,
        revision: str = "main",
        device: str = "cuda",
        half: bool = True,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        parallelize: bool = False,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(half, bool)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        self.tokenizer = self.create_auto_tokenizer(
            pretrained, revision, subfolder, tokenizer
        )
        self.model = self.create_auto_model(pretrained, revision, subfolder)
        self.model.eval()
        torch.set_grad_enabled(
            False
        )  # Turn off gradients; we're only running inference.

        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size  # todo: adaptive batch size

        # TODO: Fix multi-gpu support.
        if half:
            self.model.half()
        self._device = torch.device(device)
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
        tokenizer: transformers.PreTrainedTokenizer = None,
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
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        TODO: Handle models without sequence lengths, e.g. XLNet (https://huggingface.co/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig).
            - tokenizer.model_max_length
        """
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        # Model config has no sequence length attribute so return the tokenizer's max length.
        return self.tokenizer.model_max_length

    @property
    def batch_size(self) -> int:
        # TODO: Fix multi-gpu
        return self._batch_size  # * gpus

    @property
    def device(self):
        # TODO: Fix multi-gpu
        return self._device

    def tok_encode(self, strings: str):
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(strings, add_special_tokens=False)

    def tok_encode_batch(self, strings: str) -> torch.Tensor:
        return self.tokenizer(
            strings, padding=True, add_special_tokens=False, return_tensors="pt"
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)


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
        tokenizer: transformers.PreTrainedTokenizer = None,
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = super().create_auto_tokenizer(
            pretrained, revision, subfolder, tokenizer
        )
        tokenizer.padding_side = "left"
        return tokenizer

    def _model_call(self, inps):
        return self.model(inps)["logits"]

    def _model_generate(
        self, context, attention_mask, max_length, stopping_criteria_ids, num_fewshot
    ):
        stopping_criteria = _get_stopping_criteria(
            self.tokenizer, stopping_criteria_ids
        )
        if num_fewshot == 0:
            generations = self.model.generate(
                context,
                attention_mask=attention_mask,
                # GPT style models require the generate `max_length` arg to include the
                # the context length, so we instead set `max_new_tokens` which is the
                # number of new tokens to generate, excluding the current number of tokens.
                max_new_tokens=max_length,
                eos_token_id=self.eot_token_id,
                do_sample=False,
            )
        else:
            generations = self.model.generate(
                context,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                stopping_criteria=stopping_criteria,
                do_sample=False,
            )

        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=context.size(1)
        )


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    @property
    def max_length(self) -> int:
        """Return the sequence length of the model.
        TODO: Currently only work for T5-based models.
        """
        return 2048

    def create_auto_tokenizer(
        self,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: transformers.PreTrainedTokenizer = None,
    ) -> transformers.PreTrainedTokenizer:
        """
        TODO: Current only work for T5-based models.
        """
        tokenizer = super().create_auto_tokenizer(
            pretrained, revision, subfolder, tokenizer
        )
        tokenizer.model_max_length = self.max_length
        return tokenizer

    def loglikelihood(self, requests):
        new_reqs = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context = [f"{self.eot_token}" if len(input_) == 0 else input_ for input_ in context]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -(self.max_length - 1) :]

            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -(self.max_length - 1) :]

            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1
                    )))
            contexts, conts = utils.split_and_pad_windows(
                rolling_token_windows,
                pad_token=self.eot_token_id,
                max_seq_len=self.max_length
            )

            # Manually create BatchEncoding tensors with attention masks as 
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            context_enc = transformers.tokenization_utils_base.BatchEncoding({
                "input_ids": contexts_enc,
                "attention_mask": (contexts_enc != self.eot_token_id).long()
            })
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding({
                "input_ids": conts_enc,
                "attention_mask": (conts_enc != self.eot_token_id).long()
            })

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            rolling_token_windows_request = [((contexts, conts), context_enc, conts_enc)]
            string_nll = self._loglikelihood_tokens(rolling_token_windows_request, disable_tqdm=True)
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []
        for chunk in tqdm(requests, total=math.ceil(len(requests)), disable=disable_tqdm):
            cache_keys, inputs_tok, targets_tok = chunk
            inputs_tok = inputs_tok.to(self.device)
            targets_tok = targets_tok.to(self.device)
            outputs = self._model_call(inputs_tok, targets_tok)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tok["input_ids"],
                targets_tok["attention_mask"],
            )
            for cache_key, log_softmax, target_tok, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tok = target_tok[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tok).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tok.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))

                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return res

    def _model_call(self, inputs_tok, targets_tok):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        return self.model(**inputs_tok, labels=targets_tok["input_ids"])

    def _model_generate(
        self, context, attention_mask, max_length, stopping_criteria_ids, num_fewshot
    ):
        stopping_criteria = _get_stopping_criteria(
            self.tokenizer, stopping_criteria_ids
        )
        if num_fewshot == 0:
            generations = self.model.generate(
                context,
                attention_mask=attention_mask,
                max_length=max_length,
                eos_token_id=self.eot_token_id,
                do_sample=False,
            )
        else:
            generations = self.model.generate(
                context,
                attention_mask=attention_mask,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                do_sample=False,
            )
        return generations


# Stopping Criteria Helpers


class MultitokenEOSCriteria(transformers.StoppingCriteria):
    def __init__(self, eos_seq_id: torch.LongTensor, tokenizer):
        self.eos_seq = tokenizer.decode(eos_seq_id)
        self.eos_seq_id = eos_seq_id
        self.eos_seq_len = len(eos_seq_id) + 1
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_token_id = input_ids[0, -self.eos_seq_len :]
        last_tokens = self.tokenizer.decode(last_token_id)
        is_stopped = self.eos_seq in last_tokens
        return is_stopped


class EOSCriteria(transformers.StoppingCriteria):
    def __init__(self, eos_token_id: torch.LongTensor):
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return input_ids[0, -1] == self.eos_token_id


def _get_stopping_criteria(tokenizer, stopping_criteria_ids):
    return transformers.StoppingCriteriaList(
        [
            MultitokenEOSCriteria(stopping_criteria_ids, tokenizer),
            EOSCriteria(tokenizer.eos_token),
        ]
    )
