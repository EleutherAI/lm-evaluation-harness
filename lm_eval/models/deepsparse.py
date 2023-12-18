from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import random
import numpy
import torch

from lm_eval import utils
from lm_eval.base import BaseLM


class DeepSparseLM(BaseLM):
    # Default max sequence length setting for when no `max_length` is provided
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
        """
        Wrapper around the DeepSparse pipeline to make it compatible with the
        llm-evaluation-harness.
        """
        super().__init__()

        import deepsparse

        self._batch_size = int(batch_size)
        self._max_length = max_length or self._DEFAULT_MAX_LENGTH
        self._max_gen_toks = max_gen_toks

        # Initialize new model and tokenizer instances
        self.model = deepsparse.TextGeneration(
            model_path=pretrained,
            sequence_length=self._max_length,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
        )
        self.tokenizer = tokenizer if tokenizer else self.model.tokenizer

        self.vocab_size = self.tokenizer.vocab_size

    def _model_call(self, inps) -> torch.Tensor:
        """
        Override the _model_call method to use the DeepSparse pipeline for
        logits generation.

        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        # Encode the tokens to strings
        prompt = self.model.tokenizer.batch_decode(inps.numpy())

        # Run the model to map the prompt to logits
        out = self.model(
            prompt=prompt,
            max_new_tokens=0,
            include_prompt_logits=True,
            output_scores=True,
        )
        logits_numpy = numpy.stack([generation.score for generation in out.generations])
        return torch.from_numpy(logits_numpy)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

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

            responses = self.model(
                sequences=context,
                max_new_tokens=max_tokens,
                stop=until,
                do_sample=False,
            )

            responses = responses if type(responses) is list else [responses]

            for response in responses:
                response = response.generations[0].text
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)

        return reorder.get_original(results)

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        pass

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)
