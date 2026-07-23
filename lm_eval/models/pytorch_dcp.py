r"""
PyTorch DCP backend for TorchTitan models in lm-evaluation-harness.

This adapter loads TorchTitan models saved as model-only PyTorch DCP checkpoints directly with
``torch.distributed.checkpoint``. It mirrors the Megatron-LM backend at the
integration boundary: lm-eval builds a framework-native model, delegates
checkpoint loading to that framework, then implements the LM interface on top.

Pipeline parallelism is not supported.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


eval_logger = logging.getLogger(__name__)


def _maybe_add_torchtitan_to_path(torchtitan_path: str | None) -> None:
    torchtitan_path = torchtitan_path or os.environ.get("TORCHTITAN_PATH")
    if torchtitan_path is None:
        return
    if not os.path.isdir(torchtitan_path):
        raise FileNotFoundError(f"TorchTitan path not found: {torchtitan_path}")
    if torchtitan_path not in sys.path:
        sys.path.insert(0, torchtitan_path)


def _str_to_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


@register_model("pytorch_dcp")
class PyTorchDCPEval(LM):
    def __init__(
        self,
        checkpoint_dir: str,
        hf_assets_path: str,
        model_name: str = "llama3",
        model_flavor: str = "8B",
        torchtitan_path: str | None = None,
        attn_backend: str = "flex",
        devices: int | None = None,
        data_parallel_replicate_degree: int = 1,
        data_parallel_shard_degree: int = -1,
        tensor_parallel_degree: int = 1,
        context_parallel_degree: int = 1,
        expert_parallel_degree: int = 1,
        spmd_backend: str = "default",
        seq_length: int = 4096,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        super().__init__()

        _maybe_add_torchtitan_to_path(torchtitan_path)

        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        if not os.path.isfile(os.path.join(checkpoint_dir, ".metadata")):
            raise FileNotFoundError(
                "PyTorch DCP checkpoint must contain a .metadata "
                f"file: {checkpoint_dir}"
            )

        self._max_length = seq_length
        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)
        self._checkpoint_dir = checkpoint_dir

        self._initialize_distributed()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if devices is not None and devices != world_size:
            raise ValueError(
                f"devices={devices} does not match torch distributed world size "
                f"{world_size}."
            )

        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        self._model, self.tokenizer, self.parallel_dims = self._build_and_load_model(
            checkpoint_dir=checkpoint_dir,
            hf_assets_path=hf_assets_path,
            model_name=model_name,
            model_flavor=model_flavor,
            attn_backend=attn_backend,
            data_parallel_replicate_degree=data_parallel_replicate_degree,
            data_parallel_shard_degree=data_parallel_shard_degree,
            tensor_parallel_degree=tensor_parallel_degree,
            context_parallel_degree=context_parallel_degree,
            expert_parallel_degree=expert_parallel_degree,
            spmd_backend=spmd_backend,
            dtype=dtype,
        )
        self._model.eval()

        self._global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._rank, self._world_size, self._eval_group = self._lm_eval_rank_info(
            data_parallel_replicate_degree=data_parallel_replicate_degree,
            data_parallel_shard_degree=self.parallel_dims.dp_shard,
            tensor_parallel_degree=tensor_parallel_degree,
            context_parallel_degree=context_parallel_degree,
            expert_parallel_degree=expert_parallel_degree,
        )

        eval_logger.info(
            "Loaded PyTorch DCP checkpoint from %s with lm-eval rank/world_size %s/%s",
            checkpoint_dir,
            self._rank,
            self._world_size,
        )

    def _initialize_distributed(self) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(
            device_type, local_rank if device_type == "cuda" else 0
        )

        if dist.is_initialized():
            return

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    def _build_and_load_model(
        self,
        *,
        checkpoint_dir: str,
        hf_assets_path: str,
        model_name: str,
        model_flavor: str,
        attn_backend: str,
        data_parallel_replicate_degree: int,
        data_parallel_shard_degree: int,
        tensor_parallel_degree: int,
        context_parallel_degree: int,
        expert_parallel_degree: int,
        spmd_backend: str,
        dtype: str,
    ):
        from torchtitan.components.checkpoint import ModelWrapper
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
        from torchtitan.distributed import ParallelDims
        from torchtitan.distributed import utils as dist_utils
        from torchtitan.tools import utils as titan_utils

        if spmd_backend not in ("default", "full_dtensor", "spmd_types"):
            raise ValueError(f"Unsupported spmd_backend: {spmd_backend}")

        parallelism = ParallelismConfig(
            data_parallel_replicate_degree=data_parallel_replicate_degree,
            data_parallel_shard_degree=data_parallel_shard_degree,
            tensor_parallel_degree=tensor_parallel_degree,
            context_parallel_degree=context_parallel_degree,
            expert_parallel_degree=expert_parallel_degree,
            pipeline_parallel_degree=1,
            spmd_backend=spmd_backend,
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        parallel_dims = ParallelDims.from_config(parallelism, world_size)
        use_parallel_model = any(
            degree > 1
            for degree in (
                parallel_dims.dp_shard,
                parallel_dims.cp,
                parallel_dims.tp,
                parallel_dims.ep,
            )
        )
        if use_parallel_model:
            parallel_dims.build_mesh()
            dist_utils.set_spmd_backend(spmd_backend)

        model_module = importlib.import_module(f"torchtitan.models.{model_name}")
        model_spec = model_module.model_registry(model_flavor, attn_backend=attn_backend)
        model_config = model_spec.model
        model_config.update_from_config(
            config=SimpleNamespace(parallelism=parallelism),
        )

        target_dtype = _str_to_torch_dtype(dtype)
        with torch.device("meta"), titan_utils.set_default_dtype(target_dtype):
            model = model_config.build()
        model.verify_module_protocol()

        if use_parallel_model:
            param_dtype = "float32" if dtype == "float32" else "bfloat16"
            model = model_spec.parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=TrainingConfig(
                    dtype=param_dtype,
                    mixed_precision_param=param_dtype,
                ),
                parallelism=parallelism,
                compile_config=CompileConfig(enable=False),
                ac_config=None,
                dump_folder=".",
            )

        model.to_empty(device=self._device)
        with torch.no_grad():
            model.init_weights(buffer_device=self._device)

        state_dict = ModelWrapper(model).state_dict()
        dcp.load(state_dict, checkpoint_id=checkpoint_dir)
        model.load_state_dict(state_dict, strict=False)

        tokenizer = HuggingFaceTokenizer(tokenizer_path=hf_assets_path)
        return model, tokenizer, parallel_dims

    def _lm_eval_rank_info(
        self,
        *,
        data_parallel_replicate_degree: int,
        data_parallel_shard_degree: int,
        tensor_parallel_degree: int,
        context_parallel_degree: int,
        expert_parallel_degree: int,
    ) -> tuple[int, int, dist.ProcessGroup | None]:
        if not dist.is_initialized():
            return 0, 1, None

        model_parallel_degree = (
            data_parallel_shard_degree
            * tensor_parallel_degree
            * context_parallel_degree
            * expert_parallel_degree
        )
        pure_replicated_dp = model_parallel_degree == 1
        if pure_replicated_dp and data_parallel_replicate_degree > 1:
            return dist.get_rank(), dist.get_world_size(), dist.group.WORLD

        # Conservative first version: any sharded/model-parallel layout is one
        # logical lm-eval worker. All ranks receive the same requests so all
        # model collectives run in the same order.
        return 0, 1, None

    @property
    def eot_token_id(self) -> int:
        eos_id = getattr(self.tokenizer, "eos_id", None)
        if eos_id is not None:
            return eos_id
        return getattr(self.tokenizer, "bos_id", 0) or 0

    @property
    def prefix_token_id(self) -> int:
        bos_id = getattr(self.tokenizer, "bos_id", None)
        return bos_id if bos_id is not None else self.eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._eval_group is None or self.world_size == 1:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self._eval_group)
        return torch.cat([t.reshape(-1) for t in gathered])

    def gather_object(self, obj: Any, dst: int = 0):
        if self._eval_group is None or self.world_size == 1:
            return [obj]
        result = [None] * self.world_size if self.rank == dst else None
        dist.gather_object(
            obj=obj,
            object_gather_list=result,
            dst=dst,
            group=self._eval_group,
        )
        return result

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def tok_encode(self, string: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(string, add_bos=add_special_tokens, add_eos=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        return context_enc, whole_enc[len(context_enc) :]

    def _model_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(input_ids.shape[0], -1)

        attention_masks = None
        inner_attention = getattr(
            getattr(self._model.config, "first_attention", None),
            "inner_attention",
            None,
        )
        if inner_attention is not None:
            from torchtitan.models.common.attention import FlexAttention, VarlenAttention

            if isinstance(
                inner_attention,
                (FlexAttention.Config, VarlenAttention.Config),
            ):
                attention_masks = self._model.get_attention_masks(positions=positions)

        with torch.inference_mode():
            return self._model(
                input_ids,
                positions=positions,
                attention_masks=attention_masks,
            )

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                continuation_enc = self.tok_encode(
                    continuation, add_special_tokens=False
                )
                if continuation_enc and continuation_enc[0] == self.prefix_token_id:
                    context_enc = continuation_enc[:1]
                    continuation_enc = continuation_enc[1:]
                else:
                    context_enc = [self.prefix_token_id]
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(n=1, batch_fn=None)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm or (self._global_rank != 0),
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            cache_key, context_enc, continuation_enc = chunk[0]
            inp = (context_enc + continuation_enc)[-(self.max_length) :]
            ctxlen = len(context_enc) - max(
                0, len(context_enc) + len(continuation_enc) - self.max_length
            )
            contlen = len(continuation_enc)

            input_ids = torch.tensor([inp], dtype=torch.long, device=self.device)
            logits = self._model_forward(input_ids)
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

            start_idx = ctxlen - 1
            end_idx = ctxlen + contlen - 1
            cont_log_probs = []
            greedy_tokens = []
            for j in range(start_idx, end_idx):
                next_token = input_ids[0, j + 1].item()
                cont_log_probs.append(log_probs[0, j, next_token].item())
                greedy_tokens.append(torch.argmax(log_probs[0, j]).item())

            actual_tokens = input_ids[0, start_idx + 1 : end_idx + 1].cpu().tolist()
            answer = (sum(cont_log_probs), greedy_tokens == actual_tokens)
            res.append(answer)

            if cache_key is not None:
                self.cache_hook.add_partial("loglikelihood", cache_key, answer)
            pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[float]:
        loglikelihoods = []
        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm or (self._global_rank != 0),
            desc="Running loglikelihood_rolling requests",
        ):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=True
            )
            string_nll = sum(x[0] for x in string_nll)
            loglikelihoods.append(string_nll)
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        return loglikelihoods

    def generate_until(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[str]:
        results = []

        def _collate_gen(req):
            ctx = req.args[0]
            return -len(self.tok_encode(ctx)), ctx

        re_ord = Collator(
            requests,
            sort_fn=_collate_gen,
            group_by="gen_kwargs",
            group_fn=lambda x: x.args[1],
        )
        chunks = re_ord.get_batched(n=1, batch_fn=None)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm or (self._global_rank != 0),
            desc="Running generate_until requests",
        )

        for chunk in chunks:
            request = chunk[0]
            context, gen_kwargs = request.args
            gen_kwargs = deepcopy(gen_kwargs)
            until = gen_kwargs.pop("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.pop("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0.0)
            top_k = gen_kwargs.pop("top_k", 0)

            tokens = self.tok_encode(context)[-(self.max_length - max_gen_toks) :]
            if not tokens:
                tokens = [self.prefix_token_id]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            generated_tokens: list[int] = []

            for _ in range(max_gen_toks):
                if input_ids.shape[1] > self.max_length:
                    input_ids = input_ids[:, -self.max_length :]

                logits = self._model_forward(input_ids)
                next_token_logits = logits[:, -1, :].float()
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    if top_k > 0:
                        top_k_vals, _ = torch.topk(
                            next_token_logits,
                            min(top_k, next_token_logits.size(-1)),
                        )
                        next_token_logits = torch.where(
                            next_token_logits < top_k_vals[:, -1].unsqueeze(-1),
                            torch.full_like(next_token_logits, float("-inf")),
                            next_token_logits,
                        )
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                next_token_id = next_token.item()
                generated_tokens.append(next_token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if next_token_id == self.eot_token_id:
                    break
                generated_text = self.tok_decode(generated_tokens)
                if any(stop_seq in generated_text for stop_seq in until):
                    break

            continuation = self.tok_decode(generated_tokens)
            for stop_seq in until:
                if stop_seq in continuation:
                    continuation = continuation.split(stop_seq)[0]
                    break
            results.append(continuation)
            self.cache_hook.add_partial("generate_until", request.args, continuation)
            pbar.update(1)

        pbar.close()
        return re_ord.get_original(results)
