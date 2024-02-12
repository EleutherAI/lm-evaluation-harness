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


import os

try:
    # enabling faster model download
    import hf_transfer

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass
