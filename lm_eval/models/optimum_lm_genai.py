from importlib.util import find_spec
from pathlib import Path

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

import sys
for path in sys.path:
    print(path)

from llm_bench.python.llm_bench_utils.ov_utils import create_text_gen_model, build_ov_tokenizer
from llm_bench.python.llm_bench_utils.model_utils import get_use_case


@register_model("openvino-causal")
class OpenVINOCausalLM(HFLM):
    """
    Optimum Intel provides a simple interface to optimize Transformer models and convert them to \
    OpenVINO™ Intermediate Representation (IR) format to accelerate end-to-end pipelines on \
    Intel® architectures using OpenVINO™ runtime.

    `lm_eval --model openvino-causal --model_args pretrained=gpt2 --task wikitext`
    """

    def __init__(
        self,
        device="cpu",
        convert_tokenizer = False,
        trust_remote_code = True,
        kv_cache = False,
        **kwargs,
    ) -> None:
        self.openvino_device = device
        self.trust_remote_code = trust_remote_code,
        self.convert_tokenizer = convert_tokenizer
        self.kv_cache = kv_cache

        super().__init__(
            device=self.openvino_device,
            backend=kwargs.pop("backend", "causal"),
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=True,
        **kwargs,
    ) -> None:
        DEFAULT_LLM_BENCH_ARGS = {
            'model': '',
            'model_id': '',
            'framework': 'ov',
            'infer_count': None,
            'num_iters': 0,
            'images': None,
            'seed': 42,
            'load_config': None,
            'memory_consumption': 0,
            'batch_size': 1,
            'fuse_decoding_strategy': False,
            'make_stateful': False,
            'save_prepared_model': None,
            'num_beams': 1,
            'fuse_cache_reorder': False,
            'torch_compile_backend': 'openvino',
        }

        ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', 'CACHE_DIR': ''}
        if self.kv_cache:
            print("kv_cache enabled")
            ov_config['KV_CACHE_PRECISION'] = 'u8'
            ov_config['DYNAMIC_QUANTIZATION_GROUP_SIZE'] = '32'
        use_case, model_type = get_use_case(pretrained)

        llm_bench_args = DEFAULT_LLM_BENCH_ARGS
        llm_bench_args['model'] = pretrained
        llm_bench_args["convert_tokenizer"] = self.convert_tokenizer
        llm_bench_args["config"] =  ov_config
        llm_bench_args["use_case"] =  use_case
        llm_bench_args["trust_remote_code"] = self.trust_remote_code
        llm_bench_args["model_type"] =  model_type
        self._model, self.tokenizer, _, _, _ = create_text_gen_model(pretrained, self.openvino_device, **llm_bench_args)
