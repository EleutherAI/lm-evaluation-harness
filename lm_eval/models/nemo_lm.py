# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import pathlib
from copy import deepcopy
from typing import List, Literal

import filelock
import numpy as np
import torch
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
    simple_parse_args_string,
)


def _patch_pretrained_cfg(
    pretrained_cfg, trainer, tensor_model_parallel_size, pipeline_model_parallel_size
):
    try:
        import omegaconf
    except ModuleNotFoundError:
        raise Exception(
            "Attempted to use 'nemo_lm' model type, but package `nemo` is not installed"
            "Please install nemo following the instructions in the README: either with a NVIDIA PyTorch or NeMo container, "
            "or installing nemo following https://github.com/NVIDIA/NeMo.",
        )

    omegaconf.OmegaConf.set_struct(pretrained_cfg, True)
    with omegaconf.open_dict(pretrained_cfg):
        attributes_to_update = {
            "sequence_parallel": False,
            "activations_checkpoint_granularity": None,
            "activations_checkpoint_method": None,
            "precision": trainer.precision,
            "global_batch_size": None,
            "tensor_model_parallel_size": tensor_model_parallel_size,
            "pipeline_model_parallel_size": pipeline_model_parallel_size,
            "apply_rope_fusion": False,
        }
        for name, value in attributes_to_update.items():
            if hasattr(pretrained_cfg, name):
                pretrained_cfg[name] = value
    return pretrained_cfg


def _get_target_from_class(target_class) -> str:
    return f"{target_class.__module__}.{target_class.__name__}"


def load_model(
    model_path: str,
    trainer,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
) -> torch.nn.Module:
    try:
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
            MegatronGPTModel,
        )
        from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
    except ModuleNotFoundError:
        raise Exception(
            "Attempted to use 'nemo_lm' model type, but package `nemo` is not installed"
            "Please install nemo following the instructions in the README: either with a NVIDIA PyTorch or NeMo container, "
            "or installing nemo following https://github.com/NVIDIA/NeMo.",
        )
    model_path = pathlib.Path(model_path)

    save_restore_connector = NLPSaveRestoreConnector()
    if model_path.is_dir():
        save_restore_connector.model_extracted_dir = model_path.as_posix()
    pretrained_cfg = save_restore_connector.restore_from(
        None, model_path.as_posix(), return_config=True, trainer=trainer
    )
    if not hasattr(pretrained_cfg, "target"):
        pretrained_cfg["target"] = _get_target_from_class(MegatronGPTModel)

    pretrained_cfg = _patch_pretrained_cfg(
        pretrained_cfg,
        trainer,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    model_to_load_path = model_path
    override_config = pretrained_cfg

    module_name, class_name = override_config.target.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)

    # monkeypatch _build_tokenizer method to be process-safe
    tokenizer_lock = filelock.FileLock(f"/tmp/{model_path.name}.tokenizer.lock")

    def _synced_build_tokenizer(self):
        with tokenizer_lock:
            self._original_build_tokenizer()

    model_class._original_build_tokenizer = model_class._build_tokenizer
    model_class._build_tokenizer = _synced_build_tokenizer

    model = model_class.restore_from(
        restore_path=model_to_load_path.as_posix(),
        trainer=trainer,
        override_config_path=override_config,
        save_restore_connector=save_restore_connector,
        map_location=f"cuda:{trainer.local_rank}",
    )

    model.freeze()
    model.training = False
    try:
        # Have to turn off activations_checkpoint_method for inference
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    return model


def setup_distributed_environment(trainer):
    try:
        from nemo.utils.app_state import AppState
    except ModuleNotFoundError:
        raise Exception(
            "Attempted to use 'nemo_lm' model type, but package `nemo` is not installed"
            "Please install nemo following the instructions in the README: either with a NVIDIA PyTorch or NeMo container, "
            "or installing nemo following https://github.com/NVIDIA/NeMo.",
        )

    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    app_state = AppState()

    return app_state


@register_model("nemo_lm")
class NeMoLM(LM):
    def __init__(
        self,
        path: str,
        max_length: int = 4096,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        devices: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        precision: Literal[
            "16-mixed",
            "bf16-mixed",
            "32-true",
            "64-true",
            64,
            32,
            16,
            "64",
            "32",
            "16",
            "bf16",
        ] = "bf16",
        **kwargs,
    ):
        try:
            from nemo.collections.nlp.modules.common.text_generation_utils import (
                generate,
            )
            from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
            from pytorch_lightning.trainer.trainer import Trainer

            self.generate = generate
        except ModuleNotFoundError:
            raise Exception(
                "Attempted to use 'nemo_lm' model type, but package `nemo` is not installed"
                "Please install nemo following the instructions in the README: either with a NVIDIA PyTorch or NeMo container, "
                "or installing nemo following https://github.com/NVIDIA/NeMo.",
            )

        super().__init__()

        if (
            tensor_model_parallel_size == 1
            and pipeline_model_parallel_size == 1
            and devices > 1
        ):
            eval_logger.info(
                f"The number of data replicas for evaluation is {devices}."
            )
            eval_logger.info(f"The total number of devices is {devices}.")
            eval_logger.info(
                "No tensor parallelism or pipeline parallelism is applied."
            )

        elif tensor_model_parallel_size * pipeline_model_parallel_size == devices:
            eval_logger.info(
                f"Setting tensor parallelism to {tensor_model_parallel_size} and pipeline parallelism to {pipeline_model_parallel_size}."
            )
            eval_logger.info(f"The total number of devices is {devices}.")
            eval_logger.info("No data parallelism is applied.")

        else:
            raise ValueError(
                "Please set the product of tensor_model_parallel_size and pipeline_model_parallel_size"
                "equal to the specified number of devices."
            )

        if num_nodes > 1:
            raise ValueError(
                "A number of nodes greater than 1 is not supported yet. Please set num_nodes as 1."
            )

        trainer = Trainer(
            strategy=NLPDDPStrategy(),
            devices=devices,
            accelerator="gpu",
            num_nodes=num_nodes,
            precision=precision,
            logger=False,
            enable_checkpointing=False,
            use_distributed_sampler=False,
        )
        # Modify the following flags only for data replication
        if (
            tensor_model_parallel_size == 1
            and pipeline_model_parallel_size == 1
            and devices > 1
        ):
            self._device = torch.device(f"cuda:{trainer.global_rank}")
            self._rank = trainer.global_rank
            self._world_size = trainer.world_size
        self.model = load_model(
            path,
            trainer,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
        ).cuda()
        self.tokenizer = self.model.tokenizer
        self.app_state = setup_distributed_environment(trainer)

        self._max_length = max_length
        self._batch_size = int(batch_size)
        self._max_gen_toks = max_gen_toks

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        args = simple_parse_args_string(arg_string)
        if additional_config:
            args["batch_size"] = additional_config.get("batch_size", 1)

        return cls(**args)

    @property
    def eot_token_id(self):
        try:
            return self.tokenizer.eos_id
        except AttributeError:
            return None

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
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def accelerator(self):
        return self._Accelerator(self.world_size)

    class _Accelerator:
        def __init__(self, world_size):
            self.world_size = world_size

        def wait_for_everyone(self):
            torch.distributed.barrier()

        def gather(self, local_tensor):
            gathered_tensors = [
                torch.zeros(1, dtype=local_tensor.dtype).cuda()
                for _ in range(self.world_size)
            ]
            torch.distributed.all_gather(gathered_tensors, local_tensor)
            return torch.cat(gathered_tensors)

    def tok_encode(self, string: str):
        return self.tokenizer.text_to_ids(string)

    def tok_decode(self, tokens):
        return self.tokenizer.ids_to_text(tokens)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
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
                rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            ctxlens = []
            contlens = []

            for _, context_enc, continuation_enc in chunk:
                # Leave one token for generation. Tokens_to_generate = 0 breaks NeMo.
                inp = (context_enc + continuation_enc)[-(self.max_length - 1) :]

                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length - 1)
                )
                ctxlens.append(ctxlen)
                contlens.append(len(continuation_enc))

                inps.append(self.tok_decode(inp))

            output = self.generate(
                self.model,
                inputs=inps,
                tokens_to_generate=1,
                min_tokens_to_generate=1,
                compute_logprob=True,
                all_probs=True,
            )

            batch_token_ids = np.asarray(output["token_ids"])[:, :-1]
            batch_logprobs = output["logprob"][:, :-1]
            batch_full_logprob = output["full_logprob"][:, :-1, :]

            # Compute greedy tokens for entire batch rather than calling it with proper ctxlen for each sample.
            # Additional tokens for each sample will be trimmed later.
            min_ctxlen = min(ctxlens)

            # Use min_ctxlen-1 instead of min_ctxlen since full_logprobs are not returns for the first token.
            batch_greedy_tokens = (
                torch.argmax(batch_full_logprob[:, min_ctxlen - 1 :, :], -1)
                .cpu()
                .numpy()
            )

            for token_ids, greedy_tokens, logprobs, ctxlen, contlen, (
                cache_key,
                _,
                _,
            ) in zip(
                batch_token_ids,
                batch_greedy_tokens,
                batch_logprobs,
                ctxlens,
                contlens,
                chunk,
            ):
                # Trim at contlen since shorter contexts in a batch will have more than one token generated.
                # Use ctxlen-1 instead of ctxlen same as for full_logprob in batch_greedy_tokens calculation
                logprobs = (logprobs[ctxlen - 1 :])[:contlen]
                logprob = sum(logprobs).tolist()

                continuation_tokens = (token_ids[ctxlen:])[:contlen]
                len_diff = ctxlen - min_ctxlen
                is_greedy = continuation_tokens == (greedy_tokens[len_diff:])[:contlen]
                if not isinstance(is_greedy, bool):
                    is_greedy = is_greedy.all()
                answer = (logprob, is_greedy)

                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)
                pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(self, requests):
        if not requests:
            return []
        res = []

        def get_until(req_args):
            until = req_args.get("until", [])
            until = deepcopy(until)  # prevent from modifying req_args for cache_key
            if self.tokenizer.ids_to_tokens([self.eot_token_id])[0] not in until:
                until.append(self.tokenizer.ids_to_tokens([self.eot_token_id])[0])
            return until

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ords = Collator(
            [reg.args for reg in requests], sort_fn=_collate, group_by="gen_kwargs"
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            req_args = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = get_until(req_args)
            max_gen_toks = req_args.get("max_gen_toks", self.max_gen_toks)

            remaining_length = self.max_length - max_gen_toks
            contexts = []
            for context, _ in chunk:
                encoded_context = self.tok_encode(context)
                encoded_context = encoded_context[-remaining_length:]
                contexts.append(self.tok_decode(encoded_context))

            output = self.generate(
                self.model,
                inputs=contexts,
                tokens_to_generate=max_gen_toks,
                end_strings=until,
                greedy=True,
            )

            answers = output["sentences"]

            continuations = []
            for context, answer in zip(contexts, answers):
                continuations.append(answer[len(context) :])

            for term in until:
                continuations = [answer.split(term)[0] for answer in continuations]

            for request, answer in zip(chunk, continuations):
                self.cache_hook.add_partial("greedy_until", request, answer)
                res.append(answer)

        return re_ords.get_original(res)
