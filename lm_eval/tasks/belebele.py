"""
The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants

https://arxiv.org/abs/2308.16884

We present Belebele, a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. Significantly expanding the language coverage of natural language understanding (NLU) benchmarks, this dataset enables the evaluation of text models in high-, medium-, and low-resource languages. Each question is based on a short passage from the Flores-200 dataset and has four multiple-choice answers. The questions were carefully curated to discriminate between models with different levels of general language comprehension. The English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. We use this dataset to evaluate the capabilities of multilingual masked language models (MLMs) and large language models (LLMs). We present extensive results and find that despite significant cross-lingual transfer in English-centric LLMs, much smaller MLMs pretrained on balanced multilingual data still understand far more languages. We also observe that larger vocabulary size and conscious vocabulary construction correlate with better performance on low-resource languages. Overall, Belebele opens up new avenues for evaluating and analyzing the multilingual capabilities of NLP systems.
Homepage: https://huggingface.co/datasets/facebook/belebele
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@misc{bandarkar2023belebele,
      title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants}, 
      author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      year={2023},
      eprint={2308.16884},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class BelebeleBase(MultipleChoiceTask):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "facebook/belebele"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # TODO: Process the documents into a dictionary with the following keys:
        def format_example(doc, keys):
            """
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer：
            """

            question = doc["question"].strip()
            choices = "".join([f"{chr(ord(key[-1:]) - ord('1') + ord('A'))}. {doc[key]}\n" for key in keys])
            prompt = f"{question}\n{choices}Answer："
            return prompt

        keys = ["mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]
        return {
            "query": format_example(doc, keys),
            "choices": ["A", "B", "C", "D"],
            "gold": ord(doc["correct_answer_num"]) - ord("1"),
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class BelebeleFr(BelebeleBase):
    DATASET_NAME = "fra_Latn"
