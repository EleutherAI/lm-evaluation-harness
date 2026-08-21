from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)


@register_model("flexrank")
class FlexRankLM(HFLM):
    """Hugging Face backend for evaluating a selected FlexRank submodel."""

    _sweep_model: ClassVar[FlexRankLM | None] = None
    _sweep_rng_state: ClassVar[dict[str, Any] | None] = None

    @classmethod
    def create_from_arg_obj(
        cls,
        arg_dict: dict[str, Any],
        additional_config: dict[str, Any] | None = None,
    ) -> FlexRankLM:
        """Create normally, or select a profile on the active sweep model."""
        if cls._sweep_model is None:
            return super().create_from_arg_obj(arg_dict, additional_config)

        cls._sweep_model.select_profile(
            size_ratio=arg_dict.get("size_ratio"),
            compression_rate=arg_dict.get("compression_rate"),
        )
        assert cls._sweep_rng_state is not None
        restore_rng_state(cls._sweep_rng_state)
        return cls._sweep_model

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        *,
        size_ratio: float | None = None,
        compression_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Load a FlexRank checkpoint and select its evaluation profile.

        Args:
            pretrained: FlexRank checkpoint or initialized FlexRank model.
            size_ratio: Target parameter budget relative to the original model.
            compression_rate: Fraction of the original parameter budget to remove.
            **kwargs: Arguments forwarded to :class:`HFLM`.
        """
        super().__init__(pretrained=pretrained, **kwargs)
        self._place_data_parallel_model()
        self.select_profile(size_ratio=size_ratio, compression_rate=compression_rate)

    def select_profile(self, *, size_ratio=None, compression_rate=None) -> None:
        """Select a FlexRank profile and reset profile-dependent runtime state."""
        reduce_size = getattr(self.model, "reduce_size", None)
        if not callable(reduce_size):
            raise TypeError(
                "The loaded checkpoint is not a FlexRank model: its model class "
                "does not provide a callable `reduce_size` method. Load a checkpoint "
                "exported by FlexRank and pass `trust_remote_code=True`."
            )

        reduce_size(size_ratio=size_ratio, compression_rate=compression_rate)
        # A fresh process starts with no remembered automatic batch sizes.
        # Clear them when reusing this wrapper so each profile behaves likewise.
        self.batch_sizes = {}

        selected_ratio = getattr(self.model, "virtual_size_ratio", None)
        if selected_ratio is not None:
            eval_logger.info(
                "Selected FlexRank profile with size ratio %.6f", selected_ratio
            )

    def _get_accelerate_args(
        self,
        parallelize: bool | None = None,
        device_map: str | None = "auto",
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        gpus: int | None = None,
    ) -> dict:
        """Avoid dispatching FlexRank before its runtime buffers materialize."""
        accelerate_args = super()._get_accelerate_args(
            parallelize=parallelize,
            device_map=device_map,
            max_memory_per_gpu=max_memory_per_gpu,
            max_cpu_memory=max_cpu_memory,
            offload_folder=offload_folder,
            gpus=gpus,
        )
        single_device_map = isinstance(accelerate_args.get("device_map"), dict) and set(
            accelerate_args["device_map"]
        ) == {""}
        if parallelize is False or single_device_map:
            accelerate_args.pop("device_map", None)
            accelerate_args.pop("max_memory", None)
        return accelerate_args

    def _place_data_parallel_model(self) -> None:
        """Place models loaded by each Accelerate worker on its local device."""
        if hasattr(self, "accelerator"):
            self.model.to(self.device)


def run_flexrank_size_sweep(cfg: Any, execute_one: Any) -> None:
    """Evaluate FlexRank profiles independently while loading the model once."""
    import copy

    size_ratios = cfg.model_args["size_ratios"]
    if isinstance(size_ratios, str):
        size_ratios = [float(value) for value in size_ratios.split(":")]
    elif not isinstance(size_ratios, (list, tuple)):
        size_ratios = [size_ratios]

    base_model_args = dict(cfg.model_args)
    base_model_args.pop("size_ratios")
    base_metadata = dict(cfg.metadata)
    base_metadata.pop("size_ratios", None)

    _seed_rngs(cfg.seed)
    model = FlexRankLM.create_from_arg_obj(
        base_model_args | {"size_ratio": size_ratios[0]},
        {
            "batch_size": cfg.batch_size,
            "max_batch_size": cfg.max_batch_size,
            "device": cfg.device,
        },
    )
    FlexRankLM._sweep_model = model
    FlexRankLM._sweep_rng_state = _capture_rng_state()

    try:
        for size_ratio in size_ratios:
            profile_cfg = copy.copy(cfg)
            profile_cfg.model_args = base_model_args | {"size_ratio": size_ratio}
            profile_cfg.metadata = base_metadata | {"size_ratio": size_ratio}
            profile_cfg.use_cache = (
                f"{cfg.use_cache}_size_ratio_{size_ratio}"
                if cfg.use_cache is not None
                else None
            )
            execute_one(profile_cfg)
    finally:
        FlexRankLM._sweep_model = None
        FlexRankLM._sweep_rng_state = None


def _seed_rngs(seeds: list[int | None] | None) -> None:
    """Apply the same pre-model-construction seeds as simple_evaluate."""
    if not seeds:
        return

    import random

    import numpy as np

    from lm_eval.utils import set_torch_seed

    if seeds[0] is not None:
        random.seed(seeds[0])
    if seeds[1] is not None:
        np.random.seed(seeds[1])
    if seeds[2] is not None:
        set_torch_seed(seeds[2])


def _capture_rng_state() -> dict[str, Any]:
    """Capture RNG state after model construction for repeatable evaluation."""
    import random

    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the post-model-construction RNG state for one profile."""
    import random

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
