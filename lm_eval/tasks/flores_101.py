"""
The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation
https://aclanthology.org/2022.tacl-1.30/

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan,
Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 Evaluation Benchmark for
Low-Resource and Multilingual Machine Translation. Transactions of the Association for Computational Linguistics,
10:522–538.

FLORES-101 is a Many-to-Many multilingual translation benchmark dataset for 101 languages.

Github: https://github.com/facebookresearch/flores
"""
from lm_eval.api.task import PromptSourceTask, PerplexityTask
from typing import List, Tuple, Optional
import datasets
import copy
import re
import numpy as np
import promptsource.templates
from lm_eval import tasks


_CITATION = """
@article{goyal-etal-2022-flores,
    title = "The {F}lores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation",
    author = "Goyal, Naman  and
      Gao, Cynthia  and
      Chaudhary, Vishrav  and
      Chen, Peng-Jen  and
      Wenzek, Guillaume  and
      Ju, Da  and
      Krishnan, Sanjana  and
      Ranzato, Marc{'}Aurelio  and
      Guzm{\'a}n, Francisco  and
      Fan, Angela",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "10",
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.tacl-1.30",
    doi = "10.1162/tacl_a_00474",
    pages = "522--538",
}}
"""


class Flores101MT(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["dev"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["devtest"]

    def max_generation_length(self):
        return 512


class Flores101MT_fewshot_wmt_fr2en(Flores101MT):
    """
    This task is Identical to the Flores101MT task, except in the few-shot setting
    where few-shot examples are created using examples from the WMT14 French-to-English
    development set, whatever the language specified in the prompt.
    """

    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_mode: Optional[str] = None,
        prompt_template: Optional[promptsource.templates.Template] = None,
        example_separator: Optional[str] = "\n###\n",
        text_target_separator: Optional[str] = " ",
        save_examples: Optional[bool] = True,
    ):
        """
        Args:
            save_examples (bool, optional, defaults to True):
                Whether to save each example and corresponding model predictions
                to an output `dict`.
            > Few-shot prompting args
            example_separator (str, optional, defaults to '\n###\n'):
                The string that will be used to separate the few-shot examples
                from the prompt example.
                Default: '\n###\n'
                    See Webson & Pavlick (2022) https://arxiv.org/pdf/2109.01247.pdf
                    for justification of this separator.
            text_target_separator (str, optional, defaults to ' '):
                The string that will be used to separate the prompt example
                from the target text.
                NOTE: This is assumed to be some form of whitespace-only separation,
                    e.g. "\n\n", "\t", "  ", etc. Otherwise, you should update
                    the Task's `promptsource` template with the appropriate
                    separator(s).
                Example:
                    Q: Where is the Eiffel Tower located? A:{text_target_separator}Paris
        """
        assert (
            text_target_separator.isspace()
        ), f"`text_target_separator` must be whitespace only. Got: `{text_target_separator}`"
        super().__init__(data_dir, cache_dir, download_mode)
        self.prompt_template = prompt_template
        self.save_examples = save_examples
        self.example_separator = example_separator
        self.text_target_separator = text_target_separator
        self.cache_dir = cache_dir
        self.download_mode = download_mode

    def fewshot_docs(self) -> datasets.Dataset:
        """Returns a wmt dataset split"""
        return (
            "valid",
            datasets.load_dataset(
                "wmt14",
                "fr-en",
                cache_dir=self.cache_dir,
                download_mode=self.download_mode,
            )["validation"]["translation"],
        )

    def fewshot_context(
        self, doc: dict, num_fewshot: int, rng: Optional[np.random.Generator]
    ) -> Tuple[str, dict]:
        """Returns a few-shot context string made up of `num_fewshot` number of
        labeled examples, and an appended prompt example without labeling.

        :param doc: dict
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rng: numpy.random.Generator
            The pseudo-random number generator used to randomly sample few-shot examples.
        :returns: Tuple[str, dict]
            ctx: str
                The fewshot context.
            logging_info: dict
                A `dict` of logging info that can be used to identify few-shot sources.
        """
        assert (
            rng is not None
        ), "A `numpy.random.Generator` argument must be provided to `rng`"

        self.get_fewshot_template()

        if num_fewshot == 0:
            labeled_examples = ""
            fewshot_idx, fewshot_target_idx, fewshot_src = ([], [], None)
        else:
            # Construct few-shot labeled examples.
            fewshot_src, fewshot_docs = self.fewshot_docs()

            fewshot_examples, fewshot_idx = self.fewshot_examples(
                fewshot_docs, k=num_fewshot, rng=rng, prompt=doc
            )
            labeled_examples_list = []
            fewshot_target_idx = []
            for fewshot_example in fewshot_examples:
                # format the example, but use the previous context of the example
                text = self.doc_to_shot_text(fewshot_example)
                targets = self.doc_to_shot_target(fewshot_example)
                # Choose 1 random target from multi-reference targets.
                target_idx = int(rng.integers(0, len(targets)))
                target = targets[target_idx].strip()
                labeled_examples_list.append(
                    self.format_example(text, target, self.text_target_separator)
                )
                fewshot_target_idx.append(target_idx)
            labeled_examples = self.example_separator.join(labeled_examples_list)
            # Leave an extra `example_separator` right before the prompt.
            labeled_examples += self.example_separator

        prompt = self.doc_to_text(doc)
        ctx = labeled_examples + prompt
        logging_info = {
            "fewshot_idx": fewshot_idx,
            "fewshot_target_idx": fewshot_target_idx,
            "fewshot_source": fewshot_src,
            "fewshot_num": num_fewshot,
            "ctx": ctx,
        }
        return ctx, logging_info

    def doc_to_shot_text(self, doc: dict) -> str:
        text, _ = self.shot_prompt_template.apply(doc)
        return text

    def doc_to_shot_target(self, doc: dict) -> List[str]:
        _, target = self.shot_prompt_template.apply(doc)
        return target

    def fewshot_values(self):
        return "French", "English", "{{ fr }}", "{{ en }}"

    # heuristically hack the prompt template used to create few-shot examples
    def get_fewshot_template(self):
        self.shot_prompt_template = copy.deepcopy(self.prompt_template)

        # get things to replace in the prompt
        src_lang, trg_lang = self.prompt_template.name.split("-")[-2:]
        src_sent, trg_sent = re.findall("{{ .+? }}", self.prompt_template.jinja)
        # new attributes to drop in as replacement
        new_src_lang, new_trg_lang, new_src_sent, new_trg_sent = self.fewshot_values()
        # create new prompt
        assert len(re.findall(src_lang, self.shot_prompt_template.jinja)) == 1
        assert len(re.findall(trg_lang, self.shot_prompt_template.jinja)) == 1
        for old_text, new_text in [
            (src_lang, new_src_lang),
            (trg_lang, new_trg_lang),
            (src_sent, new_src_sent),
            (trg_sent, new_trg_sent),
        ]:
            self.shot_prompt_template.jinja = self.shot_prompt_template.jinja.replace(
                old_text, new_text
            )
        return self.shot_prompt_template

    def fewshot_examples(
        self,
        docs: datasets.Dataset,
        k: int,
        rng: np.random.Generator,
        prompt: dict = None,
    ) -> Tuple[List[dict], List[int]]:
        """Returns `k` random examples from the set of documents in `docs`.

        Args:
            docs (datasets.Dataset):
                The dataset of documents to sample few-shot examples from.
            k (int):
                The number of few-shot examples.
            rng (np.random.Generator):
                The pseudo-random number generator used to randomly sample examples.
            prompt (Optional[dict]):
                The prompt document. Specify this to ensure the prompt is not in
                the set of few-shot examples.

        Returns:
            A tuple of two lists. The first list contains the few-shot examples
        """
        random_indices = np.arange(len(docs)).tolist()
        rng.shuffle(random_indices)

        i = 0
        fewshot_examples, fewshot_idx = [], []
        for idx in random_indices:
            if i >= k:  # Break when we have enough examples.
                break
            is_same_prompt = False
            # is never same prompt with this task
            # is_same_prompt = prompt is not None and all(
            #    # Skips the `doc_id` key assigned to `prompt`s during eval pre-processing.
            #    docs[idx][k] == prompt[k]
            #    for k in docs[idx].keys()
            # )

            if self.invalid_doc_for_prompt(docs[idx]) or is_same_prompt:
                continue
            fewshot_examples.append(docs[idx])
            fewshot_idx.append(int(idx))
            i += 1
        return fewshot_examples, fewshot_idx


class Flores101MT_fewshot_wmt_hi2en(Flores101MT_fewshot_wmt_fr2en):
    """
    This task is Identical to the Flores101MT task, except in the few-shot setting
    where few-shot examples are created using examples from the WMT14 Hindi-to-English
    development set, whatever the language specified in the prompt.
    """

    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"

    def fewshot_docs(self) -> datasets.Dataset:
        """Returns a wmt dataset split"""
        return (
            "valid",
            datasets.load_dataset(
                "wmt14",
                "hi-en",
                cache_dir=self.cache_dir,
                download_mode=self.download_mode,
            )["validation"],
        )


class Flores101MT_fewshot_fr2en(Flores101MT):
    """
    This task is Identical to the Flores101MT task, except in the few-shot setting
    where few-shot examples are created using French as the source language and English
    as the target language, whatever the language specified in the prompt.
    """

    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"

    def fewshot_values(self):
        return "French", "English", "{{ sentence_fra }}", "{{ sentence_eng }}"

    # heuristically hack the prompt template used to create few-shot examples
    def get_fewshot_template(self):
        self.shot_prompt_template = copy.deepcopy(self.prompt_template)

        # get things to replace in the prompt
        src_lang, trg_lang = self.prompt_template.name.split("-")[-2:]
        src_sent, trg_sent = re.findall("{{ .+? }}", self.prompt_template.jinja)
        # new attributes to drop in as replacement
        new_src_lang, new_trg_lang, new_src_sent, new_trg_sent = self.fewshot_values()
        # create new prompt
        assert len(re.findall(src_lang, self.shot_prompt_template.jinja)) == 1
        assert len(re.findall(trg_lang, self.shot_prompt_template.jinja)) == 1
        for old_text, new_text in [
            (src_lang, new_src_lang),
            (trg_lang, new_trg_lang),
            (src_sent, new_src_sent),
            (trg_sent, new_trg_sent),
        ]:
            self.shot_prompt_template.jinja = self.shot_prompt_template.jinja.replace(
                old_text, new_text
            )
        return self.shot_prompt_template

    def fewshot_context(
        self, doc: dict, num_fewshot: int, rng: Optional[np.random.Generator]
    ) -> Tuple[str, dict]:
        """Returns a few-shot context string made up of `num_fewshot` number of
        labeled examples, and an appended prompt example without labeling.

        :param doc: dict
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rng: numpy.random.Generator
            The pseudo-random number generator used to randomly sample few-shot examples.
        :returns: Tuple[str, dict]
            ctx: str
                The fewshot context.
            logging_info: dict
                A `dict` of logging info that can be used to identify few-shot sources.
        """
        assert (
            rng is not None
        ), "A `numpy.random.Generator` argument must be provided to `rng`"

        self.get_fewshot_template()

        if num_fewshot == 0:
            labeled_examples = ""
            fewshot_idx, fewshot_target_idx, fewshot_src = ([], [], None)
        else:
            # Construct few-shot labeled examples.
            fewshot_docs = self.fewshot_docs()
            fewshot_src = str(fewshot_docs.split)
            fewshot_examples, fewshot_idx = self.fewshot_examples(
                fewshot_docs, k=num_fewshot, rng=rng, prompt=doc
            )
            labeled_examples_list = []
            fewshot_target_idx = []
            for fewshot_example in fewshot_examples:
                # format the example, but use the previous context of the example
                text = self.doc_to_shot_text(fewshot_example)
                targets = self.doc_to_shot_target(fewshot_example)
                # Choose 1 random target from multi-reference targets.
                target_idx = int(rng.integers(0, len(targets)))
                target = targets[target_idx].strip()
                labeled_examples_list.append(
                    self.format_example(text, target, self.text_target_separator)
                )
                fewshot_target_idx.append(target_idx)
            labeled_examples = self.example_separator.join(labeled_examples_list)
            # Leave an extra `example_separator` right before the prompt.
            labeled_examples += self.example_separator

        prompt = self.doc_to_text(doc)
        ctx = labeled_examples + prompt
        logging_info = {
            "fewshot_idx": fewshot_idx,
            "fewshot_target_idx": fewshot_target_idx,
            "fewshot_source": fewshot_src,
            "fewshot_num": num_fewshot,
            "ctx": ctx,
        }
        return ctx, logging_info

    def doc_to_shot_text(self, doc: dict) -> str:
        text, _ = self.shot_prompt_template.apply(doc)
        return text

    def doc_to_shot_target(self, doc: dict) -> List[str]:
        _, target = self.shot_prompt_template.apply(doc)
        return target


class Flores101MT_fewshot_hi2en(Flores101MT_fewshot_fr2en):
    """
    This task is Identical to the Flores101MT task, except in the few-shot setting
    where few-shot examples are created using Hindi as the source language and English
    as the target language, whatever the language specified in the prompt.
    """

    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"

    def fewshot_values(self):
        return "Hindi", "English", "{{ sentence_hin }}", "{{ sentence_eng }}"


class Flores101MT_fewshot_fr2ar(Flores101MT_fewshot_fr2en):
    """
    This task is Identical to the Flores101MT task, except in the few-shot setting
    where few-shot examples are created using French as the source language and Arabic
    as the target language, whatever the language specified in the prompt.
    """

    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"

    def fewshot_values(self):
        return "French", "Arabic", "{{ sentence_fra }}", "{{ sentence_ara }}"


class Flores101MT_fewshot_en2bn(Flores101MT_fewshot_fr2en):
    """
    This task is Identical to the Flores101MT task, except in the few-shot setting
    where few-shot examples are created using English as the source language and Bengali
    as the target language, whatever the language specified in the prompt.
    """

    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"

    def fewshot_values(self):
        return "English", "Bengali", "{{ sentence_eng }}", "{{ sentence_ben }}"


class Flores101Perplexity(PerplexityTask):
    """Computes the perplexity for a specific language translation of Flores-101.

    NOTE: B/c promptsource provides templates from the `all` flores-101 split, we specify
    which language to use by taking the first language code in the template name
    and compute perplexity on that.

    For example, to run perplexity on English translations of Flores-101 you
    should pass in templates of the form: 'translate-this-eng-{target}'
    """

    VERSION = 0
    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"
    LANG = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["dev"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["devtest"]

    def doc_to_target(self, doc):
        """This is a null prompt task. We need to get the target from the doc."""
        lang = self._get_lang_from_template()
        target = [doc[f"sentence_{lang}"]]
        return target

    def _get_lang_from_template(self):
        template_name = self.prompt_template.name
        # Get the first language in the prompt name is the language we'll use
        # to compute perplexity on: 'translate-this-ara-asm' -> 'ara'
        lang = template_name.rsplit("-")[-2]
        return lang

    def process_results(self, doc, results):
        if self.save_examples:
            out, log = super().process_results(doc, results)
            log["topic"] = doc["topic"]
            log["domain"] = doc["domain"]
            return out, log
        else:
            return super().process_results(doc, results)


def list_templates() -> List[str]:
    """Returns a list of non-overlapping language template names in Flores-101
    which can be used to compute multi-lingual perplexity; See docstrings of
    `Flores101Perplexity`.

    Example Usage:
    ```python
    lm_eval.get_task_list(
        'flores_101_ppl',
        template_names=lm_eval.tasks.flores_101.list_templates()
    )
    ```
    """
    unique_templates = []
    templates = tasks._get_templates_from_task(Flores101Perplexity)
    for template in templates.all_template_names:
        lang = "-".join(template.rsplit("-")[:-1])
        if not any(lang in i for i in unique_templates):
            unique_templates.append(template)
    return unique_templates
