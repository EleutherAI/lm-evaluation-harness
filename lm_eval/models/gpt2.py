import transformers
import torch
import math
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
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
        ).to(self.device)
        self.gpt2.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
        )

        #assert isinstance(
        #    self.tokenizer,
        #    (
        #        transformers.GPT2Tokenizer,
        #        transformers.GPT2TokenizerFast,
        #        transformers.T5Tokenizer,
        #        transformers.T5TokenizerFast,
        #    ),
        #), "this tokenizer has not been checked for compatibility yet!"

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
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

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

    def _model_generate(self, context, max_length, eos_token_id, k=1, temperature=0., k_batch=None):
        assert (isinstance(k, int) and k >= 1), f"Incorrect number of candidates to generate: {k}"
        assert temperature >= 0., f"Negative sampling temperature: {temperature}"
        
        # Whether to sample or to decode greedily
        do_sample = (temperature != 0.)
        if not do_sample:
            # If decoding greedily, only sample once
            assert k == 1, f"Decoding greedily but {k} generations"

        if k_batch is not None and k > 1:
            context = context.expand(k_batch, context.shape[1])
            generated_vectors = [
                self.gpt2.generate(
                    context, max_length=max_length, eos_token_id=eos_token_id,
                    do_sample=do_sample, temperature=temperature
                ) for _ in range(math.ceil(k/k_batch))
            ]
            # Pad the generated vectors such that they have the same length
            max_length = max(element.size(1) for element in generated_vectors)
            padded_vectors = []
            for vector in generated_vectors:
                if vector.size(1) < max_length:
                    vector = torch.cat([vector, torch.zeros(vector.size(0), max_length-vector.size(1), dtype=torch.int32, device=self._device)], dim=1)
                padded_vectors.append(vector)
            return torch.cat(padded_vectors, dim=0)[:k]
        
        assert k_batch is None

        if k > 1:
            context = context.expand(k, context.shape[1])
        return self.gpt2.generate(
            context, max_length=max_length, eos_token_id=eos_token_id,
            do_sample=do_sample, temperature=temperature
        )


# for backwards compatibility
GPT2LM = HFLM
