import transformers
import torch
from lm_eval.base import BaseLM


def get_args_for_accelerate(pretrained, max_memory_per_gpu, max_cpu_memory, dtype=None):
    config = transformers.AutoConfig.from_pretrained(pretrained)
    max_memory_per_gpu = {i:max_memory_per_gpu for i in range(torch.cuda.device_count())}
    max_memory_per_gpu["cpu"] = max_cpu_memory
    if dtype is not None:
        torch_dtype = config.torch_dtype
    else:
        torch_dtype = getattr(torch, dtype)

    return max_memory_per_gpu, torch_dtype

class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        accelerate=False,
        max_memory_per_gpu=None,
        skip_tokenizer=False,
        offload_folder="./offload",
        max_cpu_memory=None,
        dtype=None,
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
        if not accelerate:
            self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
            ).to(self.device)
        else:
            max_memory_per_gpu, torch_dtype = get_args_for_accelerate(pretrained, max_memory_per_gpu, max_cpu_memory, dtype)
            self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
                device_map="auto",
                max_memory_per_gpu=max_memory_per_gpu,
                torch_dtype=torch_dtype,
                offload_folder=offload_folder,
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
        ) and not skip_tokenizer:
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
            return self.gpt2(inps)[0][:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.gpt2.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
GPT2LM = HFLM
