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
from typing import List
from lm_eval import tasks
from lm_eval.api.task import PerplexityTask, PromptSourceTask


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
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["devtest"]

    def max_generation_length(self):
        return 512


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
        return False

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["dev"]

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
