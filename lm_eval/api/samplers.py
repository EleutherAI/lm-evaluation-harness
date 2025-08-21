import logging
import warnings
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Iterable, Optional, Union

import datasets
from PIL import Image

if TYPE_CHECKING:
    from random import Random

    from lm_eval.api.task import ConfigurableTask, Task


def pil_image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    byte_data = buf.getvalue()
    buf.close()
    return byte_data


eval_logger = logging.getLogger("lm-eval")


class ContextSampler:
    def __init__(
        self,
        docs: list[dict],
        task: Union["Task", "ConfigurableTask"],
        fewshot_indices: Optional[Iterable] = None,
        rnd: Optional["Random"] = None,
    ) -> None:
        self.rnd = rnd
        if not self.rnd:
            raise ValueError(
                "A `random.Random` generator argument must be provided to `rnd` of FewShotSampler!"
            )

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        if (
            self.config.fewshot_config is not None
            and self.config.fewshot_config.get("doc_to_text", None) is not None
        ):
            self.doc_to_text = partial(
                self.task.doc_to_text,
                doc_to_text=self.config.fewshot_config.get("doc_to_text", None),
            )
        else:
            self.doc_to_text = self.task.doc_to_text

        if (
            self.config.fewshot_config is not None
            and self.config.fewshot_config.get("doc_to_target", None) is not None
        ):
            self.doc_to_target = partial(
                self.task.doc_to_target,
                doc_to_target=self.config.fewshot_config.get("doc_to_target", None),
            )
        else:
            self.doc_to_target = self.task.doc_to_target

        if (
            self.config.fewshot_config is not None
            and self.config.fewshot_config.get("doc_to_choice", None) is not None
        ):
            self.doc_to_choice = partial(
                self.task.doc_to_choice,
                doc_to_choice=self.config.fewshot_config.get("doc_to_choice", None),
            )
        else:
            self.doc_to_choice = self.task.doc_to_choice

        if (
            self.config.fewshot_config is not None
            and self.config.fewshot_config.get("doc_to_image", None) is not None
        ):
            self.doc_to_image = partial(
                self.task.doc_to_image,
                doc_to_image=self.config.fewshot_config.get("doc_to_image", None),
            )
        else:
            self.doc_to_image = self.task.doc_to_image

        if (
            self.config.fewshot_config is not None
            and self.config.fewshot_config.get("doc_to_audio", None) is not None
        ):
            self.doc_to_audio = partial(
                self.task.doc_to_audio,
                doc_to_audio=self.config.fewshot_config.get("doc_to_audio", None),
            )
        else:
            self.doc_to_audio = self.task.doc_to_audio

        if (
            self.config.fewshot_config is not None
            and self.config.fewshot_config.get("doc_to_video", None) is not None
        ):
            self.doc_to_audio = partial(
                self.task.doc_to_video,
                doc_to_audio=self.config.fewshot_config.get("doc_to_video", None),
            )
        else:
            self.doc_to_video = self.task.doc_to_video

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            if not isinstance(self.docs, datasets.Dataset):
                raise ValueError(
                    "Got `fewshot_indices` but fewshot_docs are not a HF dataset. Don't use both `fewshot_indices` and a user-defined few-shot sample list simultaneously"
                )
            self.docs = self.docs.select(fewshot_indices)

    def _select_n_new_docs(self, doc: dict, num_fewshot: int):
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )

        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        return selected_docs

    def update_multimodal_args(self, multimodal_args, doc):
        if self.config.doc_to_image:
            multimodal_args.setdefault("visual", []).extend(self.doc_to_image(doc))
        if self.config.doc_to_audio:
            multimodal_args.setdefault("audio", []).extend(self.doc_to_audio(doc))
        if self.config.doc_to_video:
            multimodal_args.setdefault("video", []).extend(self.doc_to_video(doc))
        return multimodal_args

    def update_user_content(self, user_content, doc=None, images=None, audios=None, videos=None):
        def add_image(image):
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": image
                }
            )

        def add_audio(audio):
            user_content.append(
                {
                    "type": "audio_url",
                    "audio_url": audio,
                }
            )

        def add_video(video):
            user_content.append(
                {
                    "type": "video_url",
                    "video_url": video,
                }
            )

        if self.config.doc_to_image and doc:
            for image in self.doc_to_image(doc):
                add_image(image)
        if images:
            for image in images:
                add_image(image)
        if self.config.doc_to_audio and doc:
            for audio in self.doc_to_audio(doc):
                add_audio(audio)
        if audios:
            for audio in audios:
                add_audio(audio)
        if self.config.doc_to_video and doc:
            for video in self.doc_to_video(doc):
                add_video(video)
        if videos:
            for video in videos:
                add_video(video)

        return user_content

    def get_context(self, doc: dict, num_fewshot: int, gen_prefix: str = None):
        prefix = gen_prefix + " " if gen_prefix else ""
        selected_docs = self._select_n_new_docs(doc, num_fewshot)

        labeled_examples = ""
        multimodal_args = {}

        for doc in selected_docs:
            doc_content = self.doc_to_text(doc)
            doc_target = self.doc_to_target(doc)
            if self.config.doc_to_choice is None or isinstance(doc_content, str):
                labeled_examples += doc_content
            else:
                labeled_examples += self.doc_to_choice(doc)[doc_content]

            if doc_target != "":
                if self.target_delimiter.isspace() and str(doc_target)[0].isspace():
                    # TODO: add logger warn once here.
                    warnings.warn(
                        "Both target_delimiter and target start with a space. This may cause issues.",
                        Warning,
                        stacklevel=2,
                    )
                labeled_examples += self.target_delimiter
                labeled_examples += prefix
                labeled_examples += (
                    str(doc_target[0])
                    if isinstance(doc_target, list)
                    else doc_target
                    if self.config.doc_to_choice is None or isinstance(doc_target, str)
                    else str(self.doc_to_choice(doc)[doc_target])
                )
                labeled_examples += self.fewshot_delimiter
            multimodal_args = self.update_multimodal_args(multimodal_args, doc)
        return labeled_examples, multimodal_args

    def get_chat_context(
        self,
        doc: dict,
        num_fewshot: int,
        pass_multimodal_args_to_chat_history: bool = False,
        fewshot_as_multiturn: bool = False,
        gen_prefix: Optional[str] = None,
    ):
        # TODO: Do we need any other delimiter
        prefix = gen_prefix + " " if gen_prefix else ""
        chat_history = []
        multimodal_args = {}
        selected_docs = self._select_n_new_docs(doc, num_fewshot)

        if fewshot_as_multiturn:
            for doc in selected_docs:
                doc_content = self.doc_to_text(doc)
                doc_target = self.doc_to_target(doc)
                user_text = doc_content \
                    if self.config.doc_to_choice is None \
                    or isinstance(doc_content, str) \
                    else self.doc_to_choice(doc)[doc_content]
                assistant_text = prefix + str(doc_target[0]) \
                    if isinstance(doc_target, list) \
                    else prefix + doc_target \
                    if self.config.doc_to_choice is None \
                    or isinstance(doc_target, str) \
                    else prefix + str(self.doc_to_choice(doc)[doc_target])
                if pass_multimodal_args_to_chat_history:
                    user_content = [
                        {
                            "type": "text",
                            "text": user_text,
                        }
                    ]
                    user_content = self.update_user_content(user_content, doc)
                    assistant_content = [
                        {
                            "type": "text",
                            "text": assistant_text,
                        }
                    ]
                else:
                    user_content = user_text
                    multimodal_args = self.update_multimodal_args(multimodal_args, doc)
                    assistant_content = assistant_text

                chat_history.append(
                    {
                        "role": "user",
                        "content": user_content,
                    }
                )
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    }
                )
        else:
            # get fewshot context as one user turn
            labeled_examples, multimodal_args = self.get_context(doc, num_fewshot, gen_prefix=gen_prefix)
            user_content = [
                {
                    "type": "text",
                    "text": labeled_examples,
                },
            ]

            if pass_multimodal_args_to_chat_history:
                user_content = self.update_user_content(
                    user_content,
                    images=multimodal_args.get("visuals"),
                    audios=multimodal_args.get("audios")
                )
                multimodal_args = {}

            chat_history.append(
                {
                    "role": "user",
                    "content": user_content
                }
            )

        return chat_history, multimodal_args

    def sample(self, n: int):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.rnd.sample(self.docs, n)


class FirstNSampler(ContextSampler):
    def sample(self, n: int) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.docs), (
            f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        )
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n: int) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, n: int) -> None:
        """ """
        pass


SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name: str):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        )
