"""
Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset
https://aclanthology.org/2020.acl-main.459.pdf

Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset. Critical Role is an unscripted, live-streamed show where a fixed group of people play Dungeons and Dragons, an open-ended role-playing game. The dataset is collected from 159 Critical Role episodes transcribed to text dialogues, consisting of 398,682 turns. It also includes corresponding abstractive summaries collected from the Fandom wiki. The dataset is linguistically unique in that the narratives are generated entirely through player collaboration and spoken interaction. For each dialogue, there are a large number of turns, multiple abstractive summaries with varying levels of detail, and semantic ties to the previous dialogues.

Homepage: https://github.com/RevanthRameshkumar/CRD3
"""
from lm_eval.api.task import PromptSourceTask


# TODO: Add the BibTeX citation for the task.
_CITATION = """
@inproceedings{
title = {Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset},
author = {Rameshkumar, Revanth  and Bailey, Peter},
year = {2020},
publisher = {Association for Computational Linguistics},
conference = {ACL}
}
"""


class CRD3(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "shanya/crd3"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

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
        return None
