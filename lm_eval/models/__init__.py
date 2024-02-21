from . import huggingface
from . import openai_completions
from . import textsynth
from . import dummy
from . import anthropic_llms
from . import gguf
from . import vllm_causallms
from . import mamba_lm
from . import optimum_lm
from . import neuron_optimum
# TODO: implement __all__


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
