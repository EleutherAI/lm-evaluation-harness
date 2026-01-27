"""
WinML backend for lm-eval-harness with NPU/GPU/CPU support.
    
This backend leverages Windows Machine Learning (WinML) to run models on various 
hardware backends including NPUs, GPUs, and CPUs. It's particularly useful for
running inference on Windows devices with dedicated Neural Processing Units.

Example usage:
    lm_eval --model winml --model_args pretrained=path/to/onnx/model.onnx,device=npu --tasks hellaswag

Supported devices:
    - npu: Neural Processing Unit (recommended for efficiency)
    - gpu: Graphics Processing Unit
    - cpu: Central Processing Unit
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
import onnxruntime_genai as og

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)


@register_model("winml")
class WindowsML(TemplateLM):
    """
    WindowsML backend for lm-eval-harness with NPU/GPU/CPU support.

    This model class provides integration with Windows Machine Learning (WindowsML)
    to enable evaluation on NPUs and other Windows-optimized hardware.
    """
    
    _DEFAULT_MAX_LENGTH = 2048

    @classmethod
    def create_from_arg_obj(
        cls, arg_dict: Dict[str, Any], additional_config: Optional[Dict[str, Any]] = None
    ) -> "WindowsML":
        """
        Override to properly merge dictionaries and avoid duplicate keyword arguments.
        
        Args:
            arg_dict: Dictionary containing model arguments
            additional_config: Optional dictionary containing additional configuration
            
        Returns:
            Instance of WindowsML class
        """
        # Merge the dictionaries, with additional_config taking precedence
        merged_config = {**(arg_dict or {})}
        if additional_config:
            # Filter out None values and merge
            filtered_additional = {k: v for k, v in additional_config.items() if v is not None}
            merged_config.update(filtered_additional)
        
        return cls(**merged_config)

    def __init__(
        self,
        pretrained: str,
        device: str = "cpu",
        max_length: Optional[int] = 4096,
        batch_size: int = 1,
        max_batch_size: int = 64,
    ) -> None:
        """
        Initialize WindowsML model.
        
        Args:
            pretrained: Path to ONNX model file or directory containing model files
            device: Target device ('npu', 'gpu', 'cpu')
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            max_batch_size: Maximum batch size for auto-batching
        """
        super().__init__()
        
        # Validate and import dependencies
        self._validate_dependencies()
        
        # Store configuration
        self.pretrained = pretrained
        self.device = device.lower()
        self.max_length = max_length or self._DEFAULT_MAX_LENGTH
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        
        self._fix_winrt_runtime()

        # Initialize Windows ML execution providers
        self._register_winml_providers_to_genai()
        
        # Setup device and execution providers using new Windows ML API
        self._setup_winml_devices_and_providers()
        
        # Load and compile ONNX model
        self._load_and_compile_model(pretrained)
        
        eval_logger.info(f"Available EP devices: {len(self.ep_device_map)} execution providers")

    def _validate_dependencies(self) -> None:
        """
        Validate that required dependencies are available.
        
        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            import onnxruntime_genai as og
            self.og = og
            eval_logger.info(f"ONNX Runtime GenAI version: {og.__version__}")
        except ImportError as e:
            raise ImportError(
                "ONNX Runtime GenAI is required for WinML backend. "
                "Install with: pip install onnxruntime-genai"
            ) from e
        
        # Also import regular ONNX Runtime for EP registration
        try:
            import onnxruntime as ort
            self.ort = ort
            eval_logger.info(f"ONNX Runtime version: {ort.__version__}")
        except ImportError as e:
            raise ImportError(
                "ONNX Runtime is also required for execution provider registration. "
                "Install with: pip install onnxruntime"
            ) from e

    def _fix_winrt_runtime(self):
        """
        This function removes the msvcp140.dll from the winrt-runtime package.
        So it does not cause issues with other libraries.
        """
        from importlib import metadata
        site_packages_path = Path(str(metadata.distribution('winrt-runtime').locate_file('')))
        dll_path = site_packages_path / 'winrt' / 'msvcp140.dll'
        if dll_path.exists():
            dll_path.unlink()

    def _register_winml_providers_to_genai(self) -> bool:
        """
        Register Windows ML execution providers to ONNX Runtime GenAI.
        
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
                InitializeOptions,
                initialize
            )
            import winui3.microsoft.windows.ai.machinelearning as winml
            
            with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
                catalog = winml.ExecutionProviderCatalog.get_default()
                providers = catalog.find_all_providers()
                for provider in providers:
                    provider.ensure_ready_async().get()
                    # Register to GenAI instead of regular ONNX Runtime
                    self.og.register_execution_provider_library(provider.name, provider.library_path)
                    eval_logger.info(f"Registered {provider.name} to ONNX Runtime GenAI")
            
            return True
        except ImportError as e:
            eval_logger.warning(f"Windows ML import error: {e}")
            return False
        except Exception as e:
            eval_logger.warning(f"Error registering providers to GenAI: {e}")
            return False

    def _setup_winml_devices_and_providers(self) -> None:
        """
        Setup execution providers using Windows ML device enumeration API.
        
        This method queries available devices and builds a mapping of execution providers."""
        try:
            # Get available EP devices using Windows ML API
            ep_devices = self.ort.get_ep_devices()
            self.ep_device_map = {}
            
            # Build device map
            for device in ep_devices:
                ep_name = device.ep_name
                if ep_name not in self.ep_device_map:
                    self.ep_device_map[ep_name] = []
                self.ep_device_map[ep_name].append(device)
            
            # Log available devices
            eval_logger.info("Available execution provider devices:")
            for name, devices in self.ep_device_map.items():
                eval_logger.info(f"Execution Provider: {name}")
                for device in devices:
                    try:
                        device_type = self.ort.OrtHardwareDeviceType(device.device.type).name
                        eval_logger.info(f" | Vendor: {device.ep_vendor:<16} | Device Type: {device_type:<8}")
                    except Exception:
                        eval_logger.info(f" | Vendor: {device.ep_vendor:<16} | Device Type: Unknown")
            
        except Exception as e:
            eval_logger.warning(f"Windows ML device enumeration failed: {e}")
            eval_logger.info("Falling back to legacy provider selection")
            self.ep_device_map = {}

    def _load_and_compile_model(self, model_path: str) -> None:
        """
        Load and optionally compile ONNX model with ONNX Runtime GenAI.
        
        Args:
            model_path: Path to ONNX model file or directory
            
        Raises:
            FileNotFoundError: If model path is not found or invalid
            Exception: If model loading fails
        """
        model_path = Path(model_path)
        
        # Handle different input formats
        if model_path.is_file() and model_path.suffix == '.onnx':
            input_model_path = model_path.parent  # GenAI expects directory
        elif model_path.is_dir():
            input_model_path = model_path
        else:
            raise FileNotFoundError(f"Model path {model_path} not found or invalid")
        
        # Load model using ONNX Runtime GenAI
        try:
            eval_logger.info(f"Loading model with ONNX Runtime GenAI from: {input_model_path}")
            
            # Load model and tokenizer using GenAI
            self.genai_model = self.og.Model(str(input_model_path))
            self.genai_tokenizer = self.og.Tokenizer(self.genai_model)
            
            eval_logger.info("Model and tokenizer loaded successfully with ONNX Runtime GenAI")
            
            # Store model info
            self.model_path = input_model_path
            
        except Exception as e:
            eval_logger.error(f"Failed to load model with ONNX Runtime GenAI from {input_model_path}: {e}")
            raise
    
    @property
    def eot_token_id(self) -> int:
        """
        Get the end-of-text token ID.
        
        Returns:
            End-of-text token ID from the GenAI tokenizer
        """
        # GenAI tokenizer uses eos_token_id
        return self.genai_tokenizer.eos_token_id
    
    @property
    def max_gen_toks(self) -> int:
        """
        Get the maximum number of tokens to generate.
        
        Returns:
            Maximum generation tokens (default: 4096)
        """
        return 4096

    def tok_encode(self, string: str, left_truncate_len: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        """
        Tokenize string and return token IDs.
        
        Args:
            string: Input string to tokenize
            left_truncate_len: If provided, truncate from the left to this length
            add_special_tokens: Whether to add special tokens (note: GenAI tokenizer handles this automatically)
            
        Returns:
            List of token IDs
        """
        # Use GenAI tokenizer for consistency with model inference
        encoding = self.genai_tokenizer.encode(string)

        # Handle left truncation if requested
        if left_truncate_len is not None and len(encoding) > left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            tokens: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        return self.genai_tokenizer.decode(tokens)

    def _run_genai_inference_for_full_logits(self, input_text: str) -> np.ndarray:
        """
        Run inference using ONNX Runtime GenAI to get full logits sequence.
        
        Args:
            input_text: Input text string to compute logits for
            
        Returns:
            Logits matrix of shape (seq_len, vocab_size) where logits[i] contains
            predictions for the token at position i+1 given tokens[0:i+1]
            
        Raises:
            Exception: If inference fails
        """
        try:
            # Encode input text to tokens
            input_tokens = self.genai_tokenizer.encode(input_text)
            
            if len(input_tokens) == 0:
                eval_logger.warning("No tokens to process; returning empty array")
                return np.empty((0, 0), dtype=np.float32)
            
            # Create generator and get full logits in single pass
            params = self.og.GeneratorParams(self.genai_model)
            params.set_search_options(max_length=4096, do_sample=False)
            generator = self.og.Generator(self.genai_model, params)

            # Append all tokens at once            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            generator.append_tokens(input_tokens)

            # Get FULL logits using get_output("logits") - this gives all positions!
            full_logits_tensor = generator.get_output("logits")
            logits_array = np.array(full_logits_tensor, dtype=np.float32)
            
            # Handle different tensor shapes
            if len(logits_array.shape) == 3:  # (batch_size, seq_len, vocab_size)
                logits_matrix = logits_array[0]  # Remove batch dimension
            elif len(logits_array.shape) == 2:  # (seq_len, vocab_size)
                logits_matrix = logits_array
            else:
                raise ValueError(f"Unexpected logits shape: {logits_array.shape}")
            
            eval_logger.debug(f"Full logits shape: {logits_matrix.shape} for {len(input_tokens)} input tokens")
            return logits_matrix
            
        except Exception as e:
            eval_logger.error(f"GenAI inference failed: {e}")
            raise

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        """
        Stub implementation - not used since we override loglikelihood directly.
        WindowsML uses the GenAI tokenizer and overrides loglikelihood to work
        with text inputs directly, avoiding tokenization round-trip issues.
        
        Args:
            requests: List of tokenized requests
            disable_tqdm: Whether to disable progress bar
            override_bs: Optional batch size override
            
        Returns:
            Empty list (method not used)
        """
        raise NotImplementedError(
            "WindowsML overrides loglikelihood() directly and does not use _loglikelihood_tokens()"
        )

    def loglikelihood(self, requests: List["Instance"], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood using ONNX Runtime GenAI, working directly with text.
        Handles empty context for unconditional likelihood computation.
        
        Args:
            requests: List of instances containing (context, continuation) text pairs
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of tuples containing (log_likelihood, is_greedy) for each request
        """
        results = []
        
        for request in tqdm(requests, disable=disable_tqdm, desc="Computing log-likelihoods"):
            context, continuation = request.args
            
            if len(continuation) == 0:
                results.append((0.0, True))
                continue
            
            try:
                # Build full text directly from context and continuation strings
                full_text = context + continuation
                
                # Tokenize context and full text separately to find continuation tokens
                # This is necessary because tokenization is context-dependent
                if context:
                    context_enc = self.tok_encode(context)
                    context_len = len(context_enc)
                else:
                    context_enc = []
                    context_len = 0
                
                # Tokenize the full text
                full_tokens = self.tok_encode(full_text)
                
                # The continuation tokens are everything after the context
                # (tokenizing "A+B" may give different tokens than concat(tokenize(A), tokenize(B)))
                continuation_tokens = full_tokens[context_len:]
                
                if len(continuation_tokens) == 0:
                    results.append((0.0, True))
                    continue
                
                # Get logits for the full sequence
                logits = self._run_genai_inference_for_full_logits(full_text)
                
                # Extract logits for continuation positions
                # logits[i] predicts token[i+1]
                if context_len == 0:
                    # For unconditional likelihood, start from position 0
                    start_idx = 0
                else:
                    # Start from the last context position
                    start_idx = context_len - 1
                
                end_idx = start_idx + len(continuation_tokens)
                
                if start_idx >= logits.shape[0] or end_idx > logits.shape[0]:
                    results.append((0.0, False))
                    continue
                
                # Extract logits for continuation
                cont_logits = logits[start_idx:end_idx, :]
                cont_tokens = np.array(continuation_tokens)
                
                if len(cont_tokens) != cont_logits.shape[0]:
                    eval_logger.warning(
                        f"Token/logit mismatch: {len(cont_tokens)} tokens vs {cont_logits.shape[0]} logits. "
                        f"Context len: {context_len}, Full tokens: {len(full_tokens)}, Logits: {logits.shape[0]}"
                    )
                    results.append((0.0, False))
                    continue
                
                # Calculate log probabilities
                log_probs = torch.log_softmax(torch.from_numpy(cont_logits), dim=-1)
                log_likelihood = sum(log_probs[i, token] for i, token in enumerate(cont_tokens))
                
                # Check if greedy (highest probability tokens)
                greedy_tokens = torch.argmax(log_probs, dim=-1).numpy()
                is_greedy = np.array_equal(greedy_tokens, cont_tokens)
                
                results.append((float(log_likelihood), bool(is_greedy)))
                
            except Exception as e:
                eval_logger.warning(f"Failed to compute loglikelihood: {e}")
                results.append((0.0, False))
        
        return results

    def loglikelihood_rolling(self, requests: List["Instance"], disable_tqdm: bool = False) -> List[float]:
        """
        Compute rolling log-likelihood for perplexity using ONNX Runtime GenAI.
        Uses sliding windows to handle sequences longer than max_length.
        
        Args:
            requests: List of instances containing text sequences
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of sum of log-likelihood values for each request
        """
        loglikelihoods = []
        
        for request in tqdm(requests, disable=disable_tqdm, desc="Computing rolling log-likelihoods"):
            string = request.args[0]
            
            # Use sliding window approach for long sequences
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            
            # Prepare windows for loglikelihood (convert to context/continuation text pairs)
            window_requests = []
            for context_tokens, continuation_tokens in rolling_token_windows:
                context_text = self.tok_decode(context_tokens)
                continuation_text = self.tok_decode(continuation_tokens)
                window_requests.append((context_text, continuation_text))
            
            # Use loglikelihood to compute log-likelihoods for all windows
            # Create Instance objects for the windows
            from lm_eval.api.instance import Instance
            window_instances = [
                Instance(request_type="loglikelihood", doc={}, arguments=(ctx, cont), idx=0, metadata={})
                for ctx, cont in window_requests
            ]
            
            string_nll = self.loglikelihood(
                window_instances,
                disable_tqdm=True,
            )
            
            # Extract log-likelihoods (discard is_greedy boolean)
            string_nll = [x[0] for x in string_nll]
            
            # Sum all window log-likelihoods
            total_nll = sum(string_nll)
            loglikelihoods.append(total_nll)
            
            # Cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total_nll)
        
        return loglikelihoods

    def generate_until(self, requests: List["Instance"], disable_tqdm: bool = False) -> List[str]:
        """
        Generate text until stopping criteria using ONNX Runtime GenAI.
        
        Args:
            requests: List of generation requests with context and generation kwargs
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of generated text strings
        """
        if not requests:
            return []
        
        results = []
        
        for request in tqdm(requests, disable=disable_tqdm, desc="Generating text"):
            context, gen_kwargs = request.args
            
            max_gen_toks = gen_kwargs.get('max_gen_toks', self.max_gen_toks)
            until = gen_kwargs.get('until', [])
            temperature = gen_kwargs.get('temperature', 0.0)
            top_p = gen_kwargs.get('top_p', 1.0)
            top_k = gen_kwargs.get('top_k', 50)
            do_sample = gen_kwargs.get('do_sample', False)
            
            try:
                # Use GenAI generation API
                generated_text = self._run_genai_generation(
                    context, max_gen_toks, until,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample
                )
                results.append(generated_text)
                
            except Exception as e:
                eval_logger.warning(f"Generation failed for request: {e}")
                results.append("")
        
        return results
    
    def _run_genai_generation(
        self, 
        prompt: str, 
        max_tokens: int = 4096, 
        stop_sequences: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False
    ) -> str:
        """
        Run text generation using ONNX Runtime GenAI.
        
        Args:
            prompt: Input text to generate from
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences that will stop generation
            temperature: Sampling temperature (0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (if False, uses greedy decoding)
            
        Returns:
            Generated text string
        """
        try:
            # Create generator parameters
            params = self.og.GeneratorParams(self.genai_model)
            
            # Set generation parameters
            # For greedy decoding (temperature=0 or do_sample=False), disable sampling
            search_options = {
                'max_length': int(max_tokens),
                'do_sample': do_sample and temperature > 0,
            }
            
            # Add sampling parameters if do_sample is enabled
            if do_sample and temperature > 0:
                search_options['temperature'] = float(temperature)
                search_options['top_p'] = float(top_p)
                search_options['top_k'] = int(top_k)
            
            params.set_search_options(**search_options)
            
            # Create generator
            generator = self.og.Generator(self.genai_model, params)

            # Encode prompt and append tokens to the generator
            input_tokens = self.genai_tokenizer.encode(prompt)
            generator.append_tokens(input_tokens)
            
            # Generate tokens
            generated_tokens = []
            while not generator.is_done():
                generator.generate_next_token()
                
                if generator.is_done():
                    break
                    
                # Get the latest generated token
                output_tokens = generator.get_sequence(0)
                base = len(input_tokens) + len(generated_tokens)
                if len(output_tokens) > base:
                    new_token = output_tokens[base]
                    generated_tokens.append(new_token)
                    
                    # Check stopping criteria
                    if len(generated_tokens) >= max_tokens:
                        break
                    
                    # Decode current generation to check stop sequences
                    if stop_sequences:
                        current_text = self.genai_tokenizer.decode(generated_tokens)
                        if any(stop_seq in current_text for stop_seq in stop_sequences):
                            break
            
            # Decode generated tokens
            if generated_tokens:
                generated_text = self.genai_tokenizer.decode(generated_tokens)
                
                # Remove stop sequences from the end
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if generated_text.endswith(stop_seq):
                            generated_text = generated_text[:-len(stop_seq)]
                            break
                
                return generated_text
            else:
                return ""
                
        except Exception as e:
            eval_logger.error(f"GenAI generation error: {e}")
            return ""