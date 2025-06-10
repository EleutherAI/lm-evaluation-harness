import copy
import itertools
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from transformers import AutoModelForPreTraining, PreTrainedModel

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.hf_vlms import HFMultimodalLM
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    stop_sequences_criteria,
)


eval_logger = logging.getLogger(__name__)


@register_model("hf_video_llava")
class HFVideoLlava(HFMultimodalLM):
    MULTIMODAL = True
    AUTO_MODEL_CLASS = AutoModelForPreTraining

    def __init__(
        self,
        pretrained: Union[str, PreTrainedModel],
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        attn_implementation=(
            "sdpa" if torch.__version__ > "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="llava_v1",
        use_cache=True,
        truncate_context=False,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)

        self.pretrained = pretrained
        self.num_frames = num_frames
        assert num_frames == 32
        self._config = self._model.config
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def _model_multimodal_generate(self, inputs, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop,
            inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[0],
        )
        return self.model.generate(
            **inputs,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "model type `video_llava` does not support loglikelihood_rolling. Use 'hf' model type for text-only loglikelihood_rolling tasks ",
            "this is because we do not support measuring the loglikelihood a model assigns to an image.",
        )

    def loglikelihood(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "'loglikelihood' requests for model type `video_llava` are not yet tested. This feature will be enabled when a loglikelihood-based multiple-choice VQA dataset is added!"
        )

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
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
            desc="Running generate_until requests with text+video input",
        )

        re_ords = Collator(
            [reg.args for reg in requests],
            _collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs, aux_arguments = zip(*chunk)

            if not isinstance(contexts, list):
                contexts = list(contexts)
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

            # only one video
            video = [arg["video"][0] for arg in aux_arguments][0]
            if isinstance(video, torchvision.io.video_reader.VideoReader):
                video = torch.stack(
                    [frame["data"] for frame in itertools.islice(video, 5)]
                )
            total_frames = len(video)
            indices = np.arange(0, total_frames, total_frames / self.num_frames).astype(
                int
            )
            if isinstance(video, torch.Tensor):
                # torchvision videos
                clip = video[indices].numpy()
            else:
                # docord videos
                clip = video.get_batch(indices).asnumpy()

            inputs = self.processor(text=contexts, videos=clip, return_tensors="pt")
            pixel_values_videos = inputs["pixel_values_videos"]

            if pixel_values_videos.shape[1] != self.num_frames:
                empty_frames = torch.zeros(
                    (
                        1,
                        self.num_frames - pixel_values_videos.shape[1],
                        *pixel_values_videos.shape[2:],
                    ),
                    dtype=pixel_values_videos.dtype,
                )
                pixel_values_videos = torch.cat(
                    [pixel_values_videos, empty_frames], dim=1
                )
                inputs["pixel_values_videos"] = pixel_values_videos

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            context_enc = inputs["input_ids"]

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            generate_ids = self._model_multimodal_generate(inputs, stop=until, **kwargs)

            outputs = (
                self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                .split("ASSISTANT:")[-1]
                .strip()
            )

            res.append(outputs)
            pbar.update(1)

        return res
