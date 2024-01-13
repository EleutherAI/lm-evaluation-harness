from pathlib import Path
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

try:
    import optimum    
except ModuleNotFoundError: 
    raise Exception("package `optimum` is not installed. Install it via `pip install optimum[openvino] ipywidgets pillow torchaudio`")
from optimum.intel.openvino import OVModelForCausalLM

@register_model("optimum-causal")
class OptimumLM(HFLM):
    """
    ???
    """

    AUTO_MODEL_CLASS = OVModelForCausalLM
    device = "cpu"

    def __init__(
        self,
        device = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
    
 
    def _create_model(
        self,
        pretrained: str,
        revision = "main",
        dtype = "auto",
        trust_remote_code = False,
        **kwargs,
    ) -> None:

        model_kwargs = kwargs if kwargs else {}

        # export=False if pretrained is a directory that contains openvino_model.xml else True
        if Path("/pretrained/openvino_model.xml").exists():
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
        
        return None

     