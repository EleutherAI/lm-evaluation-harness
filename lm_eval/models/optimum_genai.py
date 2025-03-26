import logging
from importlib.util import find_spec
import numpy as np
from typing import List
import copy
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


@register_model("openvino_genai")
class OpenVINOGenAILM(HFLM):
    """
    Using the OpenVINO GenAI backend from optimum-intel for accelerated inference
    on Intel architectures. This leverages the OpenVINO GenAI library for optimized
    text generation.
    """

    def __init__(
        self,
        device="cpu",
        config=None,
        **kwargs,
    ) -> None:
        # Define config as a class attribute first
        self._config = config if config is not None else {}
        
        if "backend" in kwargs:
            # currently only supports causal models
            assert kwargs["backend"] == "causal", (
                "Currently, only OpenVINOGenAIModelForCausalLM is supported."
            )

        self.openvino_device = device

        super().__init__(
            device=self.openvino_device,
            backend=kwargs.pop("backend", "causal"),
            **kwargs,
        )

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

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
                "package `optimum` is not installed. Please install it via `pip install optimum-intel[openvino-genai]`"
            )
        else:
            from optimum.intel.openvino_genai.modeling_base import OpenVINOGenAIModelForCausalLM

        model_kwargs = kwargs if kwargs else {}
        
        # Pass the configuration dictionary to the model
        model_kwargs["config"] = self.config
        
        # Initialize the model with proper parameters
        self._model = OpenVINOGenAIModelForCausalLM(
            model_path=pretrained,
            device=self.openvino_device,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
    
    def loglikelihood(self, requests):
        """
        Return a fake loglikelihood value for evaluation purposes.
        OpenVINO GenAI models are focused on generation and don't support loglikelihood calculation.
        """
        eval_logger.warning(
            "OpenVINO GenAI models don't support loglikelihood calculation. Returning fake values."
        )
        
        res = []
        for request in requests:
            context, continuation = request
            # Return a fake loglikelihood value and fake is_greedy flag
            # These values are not meaningful and should not be used for actual evaluation
            fake_loglikelihood = -1.0  # Fake value
            fake_is_greedy = True      # Fake value
            res.append((fake_loglikelihood, fake_is_greedy))
        
        return res

    def loglikelihood_rolling(self, requests):
        """
        Return fake rolling loglikelihood values for evaluation purposes.
        OpenVINO GenAI models are focused on generation and don't support loglikelihood calculation.
        """
        eval_logger.warning(
            "OpenVINO GenAI models don't support loglikelihood calculation. Returning fake values."
        )
        
        res = []
        for request in requests:
            context, continuation = request
            # Return fake token loglikelihoods - one value per token in continuation
            fake_token_loglikelihoods = [-1.0] * len(continuation)  # Fake values
            res.append(fake_token_loglikelihoods)
        
        return res 

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        Generate text using OpenVINO GenAI's pipeline until a specified stopping criteria is met.
        Maintains compatibility with the HFLM implementation.
        
        Args:
            requests: List of Instance objects containing generation requests
            disable_tqdm: Whether to disable the progress bar
        
        Returns:
            List of generated strings
        """
        res = []
        
        # Create progress bar
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running OpenVINO GenAI generation requests",
        )
        
        # Process each request individually
        for request in requests:
            context, gen_kwargs = request.args
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                stop_strings = kwargs.pop("until", None)
                # Extract max_gen_toks if provided, otherwise use default
                max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)
            else:
                raise ValueError(f"Expected kwargs to be of type dict but got {type(gen_kwargs)}")
            
            try:
                # Set up generation parameters
                generation_kwargs = {
                    "max_new_tokens": max_gen_toks,
                    "stop_strings": set(stop_strings),
                }
                
                # Add any other parameters from kwargs that are supported by OpenVINO GenAI
                for k, v in kwargs.items():
                    if k not in generation_kwargs:
                        generation_kwargs[k] = v

                generated_text = self._model.generate(
                    context,
                    **generation_kwargs
                )
                    
                res.append(generated_text)
                
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generated_text)
                
            except Exception as e:
                eval_logger.error(f"Error during generation: {e}")
                res.append("")
            
            pbar.update(1)
        
        pbar.close()
        return res 