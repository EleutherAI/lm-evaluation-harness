import json
from importlib.util import find_spec
from pathlib import Path

from lm_eval import utils
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = utils.eval_logger


@register_model("openvino")
class OptimumLM(HFLM):
    """
    Optimum Intel provides a simple interface to optimize Transformer models and convert them to \
    OpenVINO™ Intermediate Representation (IR) format to accelerate end-to-end pipelines on \
    Intel® architectures using OpenVINO™ runtime.

    To use an OpenVINO config, use `--model_args ov_config` to point to a json file with an OpenVINO config:
    `lm_eval --model openvino --model_args pretrained=gpt2,ov_config=config.json --task lambada_openai`
    Example json file contents: {"INFERENCE_PRECISION_HINT": "f32", "CACHE_DIR": "model_cache"}
    """

    def __init__(
        self,
        device="cpu",
        **kwargs,
    ) -> None:
        if "backend" in kwargs:
            # optimum currently only supports causal models
            assert kwargs["backend"] == "causal", (
                "Currently, only OVModelForCausalLM is supported."
            )

        self.openvino_device = device

        super().__init__(
            device=self.openvino_device,
            backend=kwargs.pop("backend", "causal"),
            **kwargs,
        )

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=False,
        **kwargs,
    ) -> None:
        if not find_spec("optimum"):
            raise ModuleNotFoundError(
                "package `optimum` is not installed. Please install it via `pip install optimum[openvino]`"
            )
        else:
            from optimum.intel.openvino import OVModelForCausalLM

        model_kwargs = kwargs if kwargs else {}
        if "ov_config" in model_kwargs:
            if not Path(model_kwargs["ov_config"]).exists():
                raise ValueError(
                    "ov_config should point to a .json file containing an OpenVINO config"
                )
            with open(model_kwargs["ov_config"]) as f:
                model_kwargs["ov_config"] = json.load(f)
                eval_logger.info(
                    f"Using custom OpenVINO config: {model_kwargs['ov_config']}"
                )

        else:
            model_kwargs["ov_config"] = {}
        model_kwargs["ov_config"].setdefault("CACHE_DIR", "")
        if "pipeline_parallel" in model_kwargs:
            if model_kwargs["pipeline_parallel"]:
                model_kwargs["ov_config"]["MODEL_DISTRIBUTION_POLICY"] = (
                    "PIPELINE_PARALLEL"
                )
        model_file = Path(pretrained) / "openvino_model.xml"
        if model_file.exists():
            export = False
        else:
            export = True

        self._model = OVModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            export=export,
            device=self.openvino_device.upper(),
            **model_kwargs,
        )
