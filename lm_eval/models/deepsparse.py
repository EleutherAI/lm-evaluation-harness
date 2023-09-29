import torch
import deepsparse
from typing import Optional, Union
from lm_eval.base import BaseLM


class DeepSparseLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__()

        # Initialize new model and tokenizer instances
        self.model = deepsparse.Pipeline.create(
            task="text-generation",
            model_path=pretrained,
            sequence_length=max_length or _DEFAULT_MAX_LENGTH,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
        )
        self.tokenizer = tokenizer if tokenizer else self.model.tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = int(batch_size)
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        # seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        # for attr in seqlen_config_attrs:
        #     if hasattr(self.model.config, attr):
        #         return getattr(self.model.config, attr)
        # if hasattr(self.tokenizer, "model_max_length"):
        #     if self.tokenizer.model_max_length == 1000000000000000019884624838656:
        #         return self._DEFAULT_MAX_LENGTH
        #     return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cpu"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        # adaptive_batch_size = None
        # if self.batch_size == "auto":
        #     # using rolling window with maximum context
        #     print("Passed argument batch_size = auto. Detecting largest batch size")
        #     batch_size = self._detect_batch_size()
        #     print(f"Determined Largest batch size: {batch_size}")
        #     adaptive_batch_size = batch_size

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            # token_context = self.tok_encode_batch(context)

            responses = self.model(
                sequences=context,
                max_new_tokens=max_tokens,
                stop=until,
                do_sample=False,
            )

            for response in responses:
                response = response.generations[0].text
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)

        return reorder.get_original(results)

