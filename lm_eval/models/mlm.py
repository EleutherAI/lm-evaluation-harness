import collections
from copy import copy
from dataclasses import dataclass
from typing import Dict, List

import more_itertools
import transformers
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.base import BaseLM


class MaskedLM(BaseLM):
    def __init__(
        self,
        device='cuda',
        pretrained='bert-base-uncased',
        revision='main',
        subfolder=None,
        tokenizer=None,
        batch_size=1
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.lm = transformers.AutoModelForMaskedLM.from_pretrained(
            pretrained, revision=revision + ("/" + subfolder if subfolder is not None else "")
        ).to(self.device)
        self.lm.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            subfolder=subfolder)

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size

    @property
    def eot_token_id(self):
        # we use SEP because that's all we have for BERT
        return self.tokenizer.sep_token_id

    @property
    def max_length(self):
        return self.tokenizer.model_max_length

    @property
    def max_gen_toks(self):
        # This class can't generate.
        return 0

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        return self.lm(**inps).logits

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError("Masked LMs can't generate.")

    @torch.inference_mode()
    def loglikelihood(self, requests):

        @dataclass
        class ModelArgs:
            kwargs: Dict[str, torch.Tensor]
            request_index: int
            masked_token_indices: range
            targets: torch.Tensor

        model_args: List[ModelArgs] = []
        for i, (context, continuation) in enumerate(tqdm(requests, desc="Preparing requests")):
            encoded = {
                key: value.squeeze(0)
                for key, value in self.tokenizer(
                    context,
                    continuation,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True
                ).items()
            }

            # We only mask things in the continuation. We know the continuation starts because the offset
            # mappings re-start at 0.
            offset_mapping = encoded.pop("offset_mapping")
            continuation_start = len(offset_mapping) - 1
            while continuation_start > 0 and offset_mapping[continuation_start].sum() == 0:
                continuation_start -= 1
            while continuation_start > 0:
                if continuation_start == 0 or offset_mapping[continuation_start, 0] < offset_mapping[continuation_start - 1, 0]:
                    break
                continuation_start -= 1

            special_tokens_mask = encoded.pop("special_tokens_mask")
            mask_ranges = []
            range_start = continuation_start
            while special_tokens_mask[range_start] and range_start < len(special_tokens_mask):
                range_start += 1
            range_end = range_start
            for range_end in range(range_start + 1, len(special_tokens_mask)):
                if special_tokens_mask[range_end]:
                    mask_ranges.append(range(range_start, range_end))
                    range_start = range_end
                else:
                    wordpiece = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][range_end].item())
                    if not wordpiece.startswith("##"):     # Whole word masking
                        mask_ranges.append(range(range_start, range_end))
                        range_start = range_end
            if range_end > range_start:
                mask_ranges.append(range(range_start, range_end))

            for mask_range in mask_ranges:
                kwargs = copy(encoded)
                kwargs['input_ids'] = kwargs['input_ids'].clone()
                kwargs['input_ids'][mask_range] = self.tokenizer.mask_token_id
                model_args.append(ModelArgs(kwargs, i, mask_range, encoded['input_ids'][mask_range]))

        model_args.sort(key=lambda ma: -len(ma.kwargs['input_ids']))

        request_index_to_results = collections.defaultdict(lambda: [])
        for batch in more_itertools.chunked(tqdm(model_args, desc="Executing requests"), self.batch_size_per_gpu):
            # Input IDs have a different padding value, so we treat them separately.
            inputs = {
                "input_ids": pad_sequence(
                    [a.kwargs.pop('input_ids') for a in batch],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id).to(self.device)
            }
            names = list(batch[0].kwargs.keys())
            for name in names:
                inputs[name] = pad_sequence(
                    [a.kwargs.pop(name) for a in batch],
                    batch_first=True,
                    padding_value=0).to(self.device)

            logits = self._model_call(inputs)

            all_logits = F.log_softmax(logits, dim=-1).detach().cpu()
            for instance, logits in zip(batch, all_logits):
                request_index_to_results[instance.request_index].extend(
                    torch.gather(
                        logits[instance.masked_token_indices],
                        -1,
                        instance.targets.unsqueeze(-1)
                    ).squeeze(-1))

        results = [
            (i, sum(logprobs))
            for i, logprobs in request_index_to_results.items()
        ]
        results.sort()
        return [(r[1].item(),) for r in results]
