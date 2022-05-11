import abc
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
        tokenizer: str = None,
        subfolder: str = None,
        revision: str = "main",
        device: str = "cuda",
        batch_size: int = 1,
        max_gen_toks: int = 256,
        parallelize: bool = False,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        self.model = self.create_auto_model(pretrained, revision, subfolder)
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            subfolder=subfolder,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        self._max_gen_toks = max_gen_toks

        # Multithreading and batching
        self._batch_size = batch_size  # todo: adaptive batch size
        # TODO: Fix multi-gpu support.
        self._device = torch.device(device)
        if parallelize:
            self.model.parallelize()
            self._device = torch.device("cuda:0")
        else:
            self.model.to(self._device)

    def create_auto_model(
        self, pretrained: str, revision: str, subfolder: str
    ) -> transformers.AutoModel:
        """ Returns a pretrained pytorch model from a pre-trained model configuration. """
        return self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision +
            ("/" + subfolder if subfolder is not None else "")
        )

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
        """ Return the sequence length of the model. 
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        TODO: Handle models without sequence lengths, e.g. XLNet (https://huggingface.co/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetConfig).
            - tokenizer.model_max_length
        """
        seqlen_config_attrs = (
            "n_positions", "max_position_embeddings", "n_ctx")
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

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps)["logits"]

    def _model_generate(
        self, context, attention_mask, max_length, stopping_criteria_ids, num_fewshot
    ):
        stopping_criteria = _get_stopping_criteria(
            self.tokenizer, stopping_criteria_ids)
        generation_length = max_length
        # GPT style models require the generate `max_length` arg to include the
        # context length.
        max_length = max_length + context.size(1)
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
        torch.set_printoptions(profile="full")
        # print("GENERATIONS", f"{generations.shape}")
        # print("GENERATIONS", generations)        # print(out.shape)
        # print("FIXED GENERATIONS", out)

        # We need to (1) exclude the context from the generation
        # and (2) not permit additional tokens beyond the max length
        # for sentences that had shorter contexts.

        # The attention mask tracks the length of each sentence.
        mask = attention_mask.sum(1)
        fixed_generations = []
        for idx in range(generations.shape[0]):
            fixed_generations.append(
                generations[
                    # For each idx in the batch
                    idx,
                    # Index from the end of the continuation until the max length
                    mask[idx]: mask[idx] + generation_length,
                ]
            )
        out = torch.stack(fixed_generations)
        return out


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    def loglikelihood(self, requests):
        res = []
        for chunk in tqdm(
            utils.chunks(requests, self.batch_size),
            total=math.ceil(len(requests) / self.batch_size),
        ):

            inputs, targets = zip(*chunk)

            # Fill in empty encoder inputs with eos_token
            inputs = (
                f"{self.eot_token}" if len(input_) == 0 else input_ for input_ in inputs
            )

            inputs_tok = self.tokenizer(
                list(inputs),
                max_length=self.max_length,
                padding=True,
                # truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)

            for key in inputs_tok:
                inputs_tok[key] = inputs_tok[key][:, -(self.max_length - 1):]

            targets_tok = self.tokenizer(
                list(targets),
                max_length=self.max_gen_toks,
                padding=True,
                # truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)

            for key in targets_tok:
                targets_tok[key] = targets_tok[key][:, -(self.max_length - 1):]

            outputs = self._model_call(inputs_tok, targets_tok)

            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                chunk,
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
                    self.cache_hook.add_partial(
                        "loglikelihood", cache_key, answer)

                res.append(answer)

        return res

    def _model_call(self, inputs_tok, targets_tok):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(**inputs_tok, labels=targets_tok["input_ids"])

    def _model_generate(self, context, attention_mask, max_length, stopping_criteria_ids, num_fewshot):
        stopping_criteria = _get_stopping_criteria(
            self.tokenizer, stopping_criteria_ids)
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
        last_token_id = input_ids[0, -self.eos_seq_len:]
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
