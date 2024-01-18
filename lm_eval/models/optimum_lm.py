from pathlib import Path
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("optimum-causal")
class OptimumLM(HFLM):
    """
    Optimum Intel provides a simple interface to optimize Transformer models and convert them to \
    OpenVINO™ Intermediate Representation (IR) format to accelerate end-to-end pipelines on \
    Intel® architectures using OpenVINO™ runtime.
    """

    def __init__(
        self,
        device = "cpu",
        **kwargs,
    ) -> None:
        
        if "backend" in kwargs:
            # optimum currently only supports causal models.
            assert kwargs["backend"] == "causal"
        else:
            raise Exception("Please be sure your model is a `causal` model.")

        assert device == "cpu"
        super().__init__(device=device, **kwargs)

    def _create_model(
        self,
        pretrained: str,
        revision = "main",
        dtype = "auto",
        trust_remote_code = False,
        **kwargs,
    ) -> None:
        
        try:
            import optimum    
        except ModuleNotFoundError: 
            raise Exception("package `optimum` is not installed. Please install it via `pip install optimum[openvino]`")
        from optimum.intel.openvino import OVModelForCausalLM

        model_kwargs = kwargs if kwargs else {}
        model_file = Path(pretrained)/"openvino_model.xml"
        if model_file.exists():
            export = False
        else:
            export = True

        self._model = OVModelForCausalLM.from_pretrained(
            pretrained,
            revision = revision,
            trust_remote_code = trust_remote_code,
            export = export,
            device = "cpu",
            **model_kwargs,
        )

     