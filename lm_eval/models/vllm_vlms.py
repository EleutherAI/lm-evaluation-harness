import copy
import logging
from typing import Dict, List, Optional

import transformers
from more_itertools import distribute
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    replace_placeholders,
    resize_image,
    undistribute,
)
from lm_eval.models.vllm_causallms import VLLM


eval_logger = logging.getLogger(__name__)


try:
    import ray
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest  # noqa: F401
    from vllm.transformers_utils.tokenizer import get_tokenizer  # noqa: F401
except ModuleNotFoundError:
    pass


DEFAULT_IMAGE_PLACEHOLDER = "<image>"


@register_model("vllm-vlm")
class VLLM_VLM(VLLM):
    MULTIMODAL = True

    def __init__(
        self,
        pretrained: str,
        trust_remote_code: Optional[bool] = False,
        revision: Optional[str] = None,
        interleave: bool = True,
        # TODO<baber>: handle max_images and limit_mm_per_prompt better
        max_images: int = 999,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        image_max_side: Optional[int] = None,
        **kwargs,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.image_max_side = image_max_side
        if self.image_max_side and (self.image_width or self.image_height):
            raise ValueError(
                "Ambiguous config for image resize: you can not specify both "
                "image_max_side and (image_width or image_height)"
            )

        if max_images != 999:
            kwargs["limit_mm_per_prompt"] = {"image": max_images}
            eval_logger.info(f"Setting limit_mm_per_prompt[image] to {max_images}")
        super().__init__(
            pretrained=pretrained,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
        self.interleave = interleave
        self.max_images = max_images
        self.processor = transformers.AutoProcessor.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.chat_applied: bool = False

    def tok_batch_multimodal_encode(
        self,
        strings: List[str],  # note that input signature of this fn is different
        images,  # TODO: typehint on this
        left_truncate_len: int = None,
        truncation: bool = False,
    ):
        images = [img[: self.max_images] for img in images]
        # TODO<baber>: is the default placeholder always <image>?
        if self.chat_applied is False:
            strings = [
                replace_placeholders(
                    string,
                    DEFAULT_IMAGE_PLACEHOLDER,
                    DEFAULT_IMAGE_PLACEHOLDER,
                    self.max_images,
                )
                for string in strings
            ]

        outputs = []
        for x, i in zip(strings, images):
            inputs = {
                "prompt": x,
                "multi_modal_data": {"image": i},
            }
            outputs.append(inputs)
        return outputs

    def _multimodal_model_generate(
        self,
        requests: List[List[dict]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )
        if self.data_parallel_size > 1:
            # vLLM hangs if resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            @ray.remote
            def run_inference_one_model(
                model_args: dict, sampling_params, requests: List[List[dict]]
            ):
                llm = LLM(**model_args)
                return llm.generate(requests, sampling_params=sampling_params)

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            inputs = ((self.model_args, sampling_params, req) for req in requests)
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)

        if self.lora_request is not None:
            outputs = self.model.generate(
                requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
        else:
            outputs = self.model.generate(
                requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
            )
        return outputs

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt=True
    ) -> str:
        self.chat_applied = True
        if not self.interleave:
            for content in chat_history:
                c = []
                text = content["content"]

                # Count and remove image placeholders
                image_count = min(
                    self.max_images, text.count(DEFAULT_IMAGE_PLACEHOLDER)
                )
                text = text.replace(DEFAULT_IMAGE_PLACEHOLDER, "")

                # Add image entries
                for _ in range(image_count):
                    c.append({"type": "image", "image": None})

                # Add single text entry at the end
                c.append({"type": "text", "text": text})

                content["content"] = c
        else:
            for content in chat_history:
                c = []
                text = content["content"]
                expected_image_count = min(
                    self.max_images, text.count(DEFAULT_IMAGE_PLACEHOLDER)
                )
                actual_image_count = 0

                text_parts = text.split(DEFAULT_IMAGE_PLACEHOLDER)

                for i, part in enumerate(text_parts):
                    # TODO: concatenate text parts (esp. if skipping images)?
                    if part:  # Add non-empty text parts
                        c.append({"type": "text", "text": part})
                    if (
                        (i < len(text_parts) - 1) and i < self.max_images
                    ):  # Add image placeholder after each split except the last
                        c.append({"type": "image"})
                        actual_image_count += 1

                content["content"] = c

                if actual_image_count != expected_image_count:
                    raise ValueError(
                        f"Mismatch in image placeholder count. Expected: {expected_image_count}, Actual: {actual_image_count}"
                    )

        return self.processor.apply_chat_template(
            chat_history,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        if requests and len(requests[0].args) < 3:
            # Fall back to non-multimodal generation.
            return super().generate_until(requests=requests, disable_tqdm=disable_tqdm)

        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests with text+image input",
        )
        # TODO: port auto-batch sizing into this.

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(
            [reg.args for reg in requests],
            _collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        eos = self.tokenizer.decode(self.eot_token_id)
        for chunk in chunks:
            contexts, all_gen_kwargs, aux_arguments = zip(*chunk)

            visuals = [
                [
                    resize_image(
                        img, self.image_width, self.image_height, self.image_max_side
                    )
                    for img in arg["visual"]
                ]
                for arg in aux_arguments
            ]

            if not isinstance(contexts, list):
                contexts = list(
                    contexts
                )  # for Qwen2-VL, processor is unhappy accepting a tuple of strings instead of a list.
                # TODO: could we upstream this workaround to HF?

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            max_ctx_len = self.max_length - max_gen_toks

            inputs = self.tok_batch_multimodal_encode(
                contexts,
                visuals,
                left_truncate_len=max_ctx_len,
            )

            cont = self._multimodal_model_generate(
                inputs, stop=until, generate=True, max_tokens=max_gen_toks, **kwargs
            )

            for output, context in zip(cont, contexts):
                generated_text = output.outputs[0].text
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        if requests and len(requests[0].args) < 3:
            # Fall back to non-multimodal generation.
            return super().loglikelihood_rolling(requests=requests)
        raise NotImplementedError(
            "model type `vllm-vlm` does not support loglikelihood_rolling. Use 'vlm' model type for text-only loglikelihood_rolling tasks ",
            "this is because we do not support measuring the loglikelihood a model assigns to an image.",
        )
