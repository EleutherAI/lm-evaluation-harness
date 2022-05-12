import transformers
import torch
from lm_eval.base import BaseLM


class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        parallelize=False,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            self._device = torch.device(device)
        else:
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            subfolder=subfolder,
        )

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        if parallelize:
            self.gpt2.parallelize()
            self._device = torch.device("cuda:0")
        else:
            self.gpt2.to(self._device)

    @property
    def eot_token(self):
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0]

    def _get_stopping_criteria(self, stopping_criteria_ids):
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

        return transformers.StoppingCriteriaList(
            [
                MultitokenEOSCriteria(stopping_criteria_ids, self.tokenizer),
                EOSCriteria(self.tokenizer.eos_token),
            ]
        )

    def _model_generate(self, context, max_length, stopping_criteria_ids, num_fewshot):
        stopping_criteria = self._get_stopping_criteria(stopping_criteria_ids)
        max_length = max_length + context.size(1)
        if num_fewshot == 0:
            generations = self.gpt2.generate(
                context,
                max_length=max_length,
                eos_token_id=self.eot_token_id,
                do_sample=False,
            )
        else:
            generations = self.gpt2.generate(
                context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                do_sample=False,
            )

        # Remove the context from the generations
        return generations[0, context.shape[1] :]


# for backwards compatibility
GPT2LM = HFLM
