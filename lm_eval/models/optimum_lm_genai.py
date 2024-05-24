from importlib.util import find_spec
from pathlib import Path

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

import sys
for path in sys.path:
    print(path)

from llm_bench.python.utils.ov_utils import create_text_gen_model, build_ov_tokenizer
from llm_bench.python.utils.model_utils import get_use_case


@register_model("optimum-causal")
class OptimumLM_2(HFLM):
    """
    Optimum Intel provides a simple interface to optimize Transformer models and convert them to \
    OpenVINO™ Intermediate Representation (IR) format to accelerate end-to-end pipelines on \
    Intel® architectures using OpenVINO™ runtime.
    """

    def __init__(
        self,
        device="cpu",
        convert_tokenizer = False,
        trust_remote_code = True,
        kv_cache = False,
        **kwargs,
    ) -> None:
        # if "backend" in kwargs:
        #     # optimum currently only supports causal models
        #     assert (
        #         kwargs["backend"] == "causal" 
        #     ), "Currently, only OVModelForCausalLM is supported."

        self.openvino_device = device
        self.trust_remote_code = trust_remote_code,
        self.convert_tokenizer = convert_tokenizer
        self.kv_cache = kv_cache

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

        MODEL_TYPE_BY_MODEL_NAME = {
            'aquilachat-7b': 'aquila',
            'baichuan2-7b': 'baichuan',
            'baichuan2-7b-chat': 'baichuan',
            'bloomz-560m': 'bloom',
            'bloomz-7b1': 'bloom',
            'bloom-1b4-zh': 'bloom',
            'codellama-7b': 'llama',
            'codet5-base-sum': 'codet5',
            'chatglm-6b': 'chatglm',
            'chatglm2-6b': 'chatglm2',
            'chatglm2-6b-gptq': 'chatglm2',
            'chatglm3-6b': 'chatglm2',
            'chatglm3-6b-gptq': 'chatglm2',
            'dolly-v2-12b': 'dolly',
            'dolly-v2-2-8b': 'dolly',
            'dolly-v2-3b': 'dolly',
            'falcon-40b': 'falcon',
            'falcon-7b-instruct': 'falcon',
            'flan-t5-large-grammar-synthesis': 't5',
            'flan-t5-xxl': 't5',
            't5': 't5',
            'gpt-2': 'gpt',
            'gpt-j-6b': 'gpt',
            'gpt-neox-20b': 'gpt',
            'instruct-gpt-j': 'gpt',
            'llama-2-7b-chat': 'llama',
            'llama-2-7b-gptq': 'llama',
            'llama-2-13b-chat': 'llama',
            'llama-7b': 'llama',
            'mistral-7b-v0.1': 'mistral',
            'mixtral-8x7b-v0.1': 'mixtral',
            'mpt-7b-chat': 'mpt',
            'mpt-30b-chat': 'mpt',
            'open-llama-3b': 'open_llama',
            'open-llama-7b': 'open_llama',
            'open-assistant-pythia-12b': 'pythia-',
            'opt-2.7b': 'opt-',
            'orca-mini-3b': 'orca-mini',
            'pythia-12b': 'pythia-',
            'phi-2': 'phi-',
            'phi-3-mini-4k-instruct': 'phi-',
            'stablelm-3b-4e1t': 'stablelm-',
            'stablelm-3b-4e1t-gptq': 'stablelm-',
            'stablelm-7b': 'stablelm-',
            'stablelm-tuned-alpha-3b': 'stablelm-',
            'stablelm-epoch-3b-preview': 'stablelm-',
            'vicuna-7b-v1.5': 'vicuna',
            'longchat-b7': 'longchat',
            'red-pajama-incite-chat-3b-v1': 'red-pajama',
            'qwen-7b-chat': 'qwen',
            'qwen-7b-chat-gptq': 'qwen',
            'qwen-14b-chat': 'qwen',
            'xgen-7b-instruct':'xgen',
            'stable-zephyr-3b-dpo': 'zephyr',
            'zephyr-7b-beta': 'zephyr',
        }

        ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', 'CACHE_DIR': ''}
        if self.kv_cache:
            print("kv_cache enabled")
            ov_config['KV_CACHE_PRECISION'] = 'u8'
            ov_config['DYNAMIC_QUANTIZATION_GROUP_SIZE'] = '32'
        use_case, model_name = get_use_case(pretrained)

        if model_name in MODEL_TYPE_BY_MODEL_NAME.keys():
            model_type = MODEL_TYPE_BY_MODEL_NAME[model_name]
            llm_bench_args = DEFAULT_LLM_BENCH_ARGS
            llm_bench_args['model'] = pretrained
            llm_bench_args["convert_tokenizer"] = self.convert_tokenizer
            llm_bench_args["config"] =  ov_config
            llm_bench_args["use_case"] =  use_case
            llm_bench_args["trust_remote_code"] = self.trust_remote_code
            llm_bench_args["model_type"] =  model_type
            self._model, self.tokenizer, _, _ = create_text_gen_model(pretrained, self.openvino_device, **llm_bench_args)
        else:
            assert False, f"ERROR model type not defined formodel: {model_name}"
