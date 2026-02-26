from __future__ import annotations

import random
from functools import cached_property
from typing import TYPE_CHECKING

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


if TYPE_CHECKING:
    from collections.abc import Sequence

    from lm_eval.api.instance import GenInstance, LLInstance


def print_args(req: GenInstance | LLInstance):
    a, b = req.args
    print(f"arg1: {a}")
    print(f"arg2: {b}")


@register_model("dummy")
class DummyLM(LM):
    tokenizer_name = "allenai/Olmo-3-32B-Think"

    def __init__(
        self,
        *args,
        write_out: bool = False,
        raw_chats: bool = False,
        rank: int = 0,
        world_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.write_out = write_out
        self.raw_chats = raw_chats
        self._rank = rank
        self._world_size = world_size
        self._device = "cpu"

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(self, requests: Sequence[LLInstance], disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            res.append((-float(request.doc_id), request.doc_id % 2 == 0))
            if self.write_out:
                print_args(request)

        return res

    def generate_until(
        self, requests: Sequence[GenInstance], disable_tqdm: bool = False
    ):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            res.append("lol")
            if self.write_out:
                print_args(request)
            if not self.raw_chats:
                assert request.arguments[0].strip() != "", (
                    f"Expected non-empty context, got {request}"
                )

        return res

    def loglikelihood_rolling(
        self, requests: Sequence[LLInstance], disable_tqdm: bool = False
    ):
        if self.write_out:
            [print_args(request) for request in requests]
        return [(-random.random(), False) for _ in tqdm(requests, disable=disable_tqdm)]

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def all_gather(self, tensor):
        if self._world_size <= 1:
            return tensor
        import torch
        import torch.distributed as dist

        gathered = [torch.zeros_like(tensor) for _ in range(self._world_size)]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered)

    def barrier(self) -> None:
        if self._world_size <= 1:
            return
        import torch.distributed as dist

        dist.barrier()

    def apply_chat_template(
        self, chat_history: Sequence[dict[str, str]], add_generation_prompt: bool = True
    ) -> str | list[dict[str, str]]:
        if self.raw_chats:
            return list(chat_history)
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
