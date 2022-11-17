"""
DiaBLa: English-French Bilingual dialogue dataset for Machine Translation
https://link.springer.com/article/10.1007/s10579-020-09514-4

Rachel Bawden, Eric Bilinski, Thomas Lavergne and Sophie Rosset
(2021). DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues
for Machine Translation. Language Resources and Evaluation(55). Pages
635–660. Springer Verlag. 10.1007/s10579-020-09514-4.

DiaBLa is an English-French dataset for the evaluation of Machine
Translation (MT) for informal, written bilingual dialogue.  It
contains 144 spontaneous dialogues (5,700+ sentences) between native
English and French speakers, mediated by one of two neural MT systems
in a range of role-play settings. The dialogues are accompanied by
fine-grained sentence-level judgments of MT quality, produced by the
dialogue participants themselves, as well as by manually normalised
versions and reference translations produced a posteriori

Homepage: http://almanach.inria.fr/software_and_resources/custom/DiaBLa-en.html
"""
from lm_eval.api.task import PromptSourceTask
from typing import List, Tuple, Optional
import datasets
import copy
import numpy as np


_CITATION = """@article{bawden_DiaBLa:-A-Corpus-of_2021,
  author = {Bawden, Rachel and Bilinski, Eric and Lavergne, Thomas and Rosset, Sophie},
  doi = {10.1007/s10579-020-09514-4},
  title = {DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation},
  year = {2021},
  journal = {Language Resources and Evaluation},
  publisher = {Springer Verlag},
  volume = {55},
  pages = {635--660},
  url = {https://hal.inria.fr/hal-03021633},
  pdf = {https://hal.inria.fr/hal-03021633/file/diabla-lre-personal-formatting.pdf},
}
"""


class DiaBLa(PromptSourceTask):

    DATASET_PATH = "rbawden/DiaBLa"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self):
        return 512

    def invalid_doc_for_prompt(self, doc) -> bool:
        if len(self.doc_to_target(doc)) == 0 or self.doc_to_target(doc)[0] == "":
            return True
        return False


class DiaBLa_1_shot_context_same(PromptSourceTask):
    """
    This task is identical to the DiaBLa task, but in the 1-shot setting takes the
    1-shot example from the previous sentence in the dialogue if this is available
    (source sentence and MT output, in the same language direction as the direction
    of the current example). N.B. this task is not currently designed for more than
    1-shot.
    """

    DATASET_PATH = "rbawden/DiaBLa"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self):
        return 512

    def invalid_doc_for_prompt(self, doc) -> bool:
        if len(self.doc_to_target(doc)) == 0 or self.doc_to_target(doc)[0] == "":
            return True
        return False

    def doc_to_shot_text(self, doc: dict) -> str:
        text, _ = self.shot_prompt_template.apply(doc)
        return text

    def doc_to_shot_target(self, doc: dict) -> List[str]:
        _, target = self.shot_prompt_template.apply(doc)
        return target

    def fewshot_docs(self) -> datasets.Dataset:
        """
        Returns the `dataset` split that the few-shot examples should be sample
        from. This prioritizes the `train_docs` split as the few-shot example
        source, then `validation_docs`, and lastly `test_docs`.
        """
        return self.test_docs()

    # heuristically hack the current template to replace the attributes 'orig' and 'ref' by
    # the original and reference sentences of the previous sentence (if available)
    def get_fewshot_template(self):
        self.shot_prompt_template = copy.deepcopy(self.prompt_template)
        old_jinja = self.shot_prompt_template.jinja
        preamble = '{% set src_sent = ""%}'
        preamble += '{% set trg_sent = "" %}'
        preamble += "{% if dialogue_history|length > 0 %}{% if utterance_meta.lang == dialogue_history[-1].utterance_meta.lang %}{% set src_sent = dialogue_history[-1].orig %}{% set trg_sent = dialogue_history[-1].ref %}{% else %}{% set src_sent = dialogue_history[-1].ref %}{% set trg_sent = dialogue_history[-1].orig %}{% endif %}{% endif %}"
        self.shot_prompt_template.jinja = preamble + old_jinja.replace(
            "{{ orig }}", "{{ src_sent }}"
        ).replace("{{ ref }}", "{{ trg_sent }}")
        return self.shot_prompt_template

    def fewshot_examples(
        self,
        docs: datasets.Dataset,
        k: int,
        rng: np.random.Generator,
        prompt: dict = None,
    ) -> Tuple[List[dict], List[int]]:
        """Returns `k` random examples from the set of documents in `docs`.

        :param docs: datasets.Dataset
            The dataset of documents to sample few-shot examples from.
        :param k: int
            The number of few-shot examples.
        :param rng: np.random.Generator
            The pseudo-random number generator used to randomly sample examples.
        :param prompt: Optional[dict]
            The prompt document. Specify this to ensure the prompt is not in
            the set of few-shot examples.
        """
        # hack the hack to the be used so that it uses other attributes
        self.get_fewshot_template()
        return [prompt], 0

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


class DiaBLa_1_shot_context_opposite(DiaBLa_1_shot_context_same):
    """
    This task is identical to the DiaBLa task, but in the 1-shot setting takes the
    1-shot example from the previous sentence in the dialogue if this is available
    (source sentence and MT output, in the same language direction as the direction
    of the current example). N.B. this task is not currently designed for more than
    1-shot.
    """

    DATASET_PATH = "rbawden/DiaBLa"
    DATASET_NAME = None

    # heuristically hack the current template to replace the attributes 'orig' and 'ref' by
    # the original and reference sentences of the previous sentence (if available)
    def get_fewshot_template(self):
        self.shot_prompt_template = copy.deepcopy(self.prompt_template)
        old_jinja = self.shot_prompt_template.jinja
        preamble = '{% set src_sent = ""%}'
        preamble += '{% set trg_sent = "" %}'
        preamble += "{% if dialogue_history|length > 0 %}{% if utterance_meta.lang != dialogue_history[-1].utterance_meta.lang %}{% set src_sent = dialogue_history[-1].orig %}{% set trg_sent = dialogue_history[-1].ref %}{% else %}{% set src_sent = dialogue_history[-1].ref %}{% set trg_sent = dialogue_history[-1].orig %}{% endif %}{% endif %}"
        self.shot_prompt_template.jinja = preamble + old_jinja.replace(
            "{{ orig }}", "{{ src_sent }}"
        ).replace("{{ ref }}", "{{ trg_sent }}").replace(
            '{% if utterance_meta.lang == "french" %}',
            '{% if utterance_meta.lang != "french" %}',
        )
        return self.shot_prompt_template


class DiaBLa_1_shot_context_orig(DiaBLa_1_shot_context_same):
    """
    This task is identical to the DiaBLa task, but in the 1-shot setting takes the
    1-shot example from the previous sentence in the dialogue if this is available
    (source sentence and MT output, in the same language direction as the direction
    of the current example). N.B. this task is not currently designed for more than
    1-shot.
    """

    DATASET_PATH = "rbawden/DiaBLa"
    DATASET_NAME = None

    # heuristically hack the current template to replace the attributes 'orig' and 'ref' by
    # the original and reference sentences of the previous sentence (if available)
    def get_fewshot_template(self):
        self.shot_prompt_template = copy.deepcopy(self.prompt_template)
        old_jinja = self.shot_prompt_template.jinja
        preamble = '{% set src_sent = ""%}{% set trg_sent = "" %}'
        preamble += "{% if dialogue_history|length > 0 %}{% set src_sent = dialogue_history[-1].orig %}{% set trg_sent = dialogue_history[-1].ref %}{% endif %}"
        self.shot_prompt_template.jinja = preamble + old_jinja.replace(
            "{{ orig }}", "{{ src_sent }}"
        ).replace("{{ ref }}", "{{ trg_sent }}").replace(
            "utterance_meta.lang", "dialogue_history[-1].utterance_meta.lang"
        )
        return self.shot_prompt_template
