import logging
import pathlib

from typing import Optional, Union, Any

import transformers
import huggingface_hub
import torch

from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import configure_pad_token, stop_sequences_criteria
from lm_eval.api.model import CacheHook

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.models.auto import model_registry


eval_logger = logging.getLogger(__name__)


@register_model("fast_llm")
class FastLLMWrapper(HFLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str | Any,
        model_name: str = "gpt",
        checkpoint_format: str | None = None,
        attn_implementation: str = "flash_attention_2",  # other supported option is "fuse"
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        dtype: Optional[str] = None,  # other supported option is "bf16"
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        **kwargs,
    ):
        # intitialize manualy fields in base class as we do not want to call init on HFLM
        # super().__init__()
        self._rank = 0
        self._world_size = 1
        self.cache_hook = CacheHook(None)

        # some inputs which are expected in HFLM but are not used by our model
        backend = "causal"
        revision = "main"
        gguf_file = None
        delta = None
        peft = None

        self.backend = backend

        if isinstance(pretrained, str):
            # this will allow download supported models automatically from HF hubs
            pretrained = self._get_model_path(pretrained)
        else:
            # if pretrained is an object we need a path to tokenizer
            # TODO: can we get it from fast_llm object?
            assert tokenizer is not None

        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )

        # initialize our model now
        self._create_model(
            pretrained=pretrained,
            model_name=model_name,
            checkpoint_format=checkpoint_format,
            attn_implementation=attn_implementation,
            dtype=dtype,
            **kwargs,
        )

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

        self.add_bos_token = add_bos_token
        # TODO: do we support gemma models?
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS"
                " token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

    def _create_model(
        self,
        pretrained,
        model_name,
        checkpoint_format,
        attn_implementation,
        dtype,
        **kwargs,
    ):
        assert model_name in model_registry.keys()
        fml_config_class: FastLLMModelConfig = model_registry[model_name]
        hf_fml_for_causual_lm_class = fml_config_class.get_huggingface_model_class()

        if not isinstance(pretrained, str):
            # We need exact class name match here as classes are derived from one another
            assert type(pretrained).__name__ == hf_fml_for_causual_lm_class.__name__
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. "
                "Please do not launch via accelerate if passing an existing model this way."
            )
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model._config
            return

        assert checkpoint_format is not None

        fm_kwards = {}
        # TODO: make proper enum
        if attn_implementation == "flash_attention_2":
            fm_kwards["attn_implementation"] = "flash_attention_2"
        elif attn_implementation == "fuse":
            fm_kwards["attn_implementation"] = "fuse"
        else:
            raise ValueError(f"Unknown attention implemetation: {attn_implementation}")

        if dtype is not None:
            # TODO: make proper enum
            assert dtype in ("bf16")
            fm_kwards["torch_dtype"] = dtype

        self._model = hf_fml_for_causual_lm_class.from_pretrained(
            CheckpointLoadConfig(
                path=pretrained,
                format=fml_config_class.get_checkpoint_format(checkpoint_format),
            ),
            use_fm_changes=True,  # NOTE: this will be removed after generate is finalized
            **fm_kwards,
        )

        self._device = self._model.device
        self._config = self._model.config

    def _get_model_path(self, model_name: str):
        # Check if it's a valid local path
        model_name_ = pathlib.Path(model_name)
        if model_name_.is_dir():
            return model_name_.absolute()
        else:
            # Otherwise, assume it's a remote Hugging Face model and download it
            return huggingface_hub.snapshot_download(repo_id=model_name)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        # TODO: do we need no_grad for our model?
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                return self.model(
                    input_ids=inps,
                    attention_mask=attn_mask,
                    labels=labels,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                ).logits
            else:
                return self.model(
                    input_ids=inps,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=None,
                    labels=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                ).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            **generation_kwargs,
        )
