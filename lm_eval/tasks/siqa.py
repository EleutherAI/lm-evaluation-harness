"""
SOCIAL IQA: Commonsense Reasoning about Social Interactions
https://aclanthology.org/D19-1454.pdf

Social IQa: Social Interaction QA, is a question-answering benchmark for testing
social commonsense intelligence. Contrary to many prior benchmarks that focus on
physical or taxonomic knowledge, Social IQa focuses on reasoning about people’s
actions and their social implications. For example, given an action like "Jesse
saw a concert" and a question like "Why did Jesse do this?", humans can easily
infer that Jesse wanted "to see their favorite performer" or "to enjoy the music",
and not "to see what's happening inside" or "to see if it works". The actions in Social IQa
span a wide variety of social situations, and answer candidates contain both human-curated
answers and adversarially-filtered machine-generated candidates.
Social IQa contains over 37,000 QA pairs for evaluating models’ abilities to reason
about the social implications of everyday events and situations.

Homepage: https://leaderboard.allenai.org/socialiqa/submissions/get-started
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{Sap2019SocialIC,
  title={Social IQA: Commonsense Reasoning about Social Interactions},
  author={Maarten Sap and Hannah Rashkin and Derek Chen and Ronan Le Bras and Yejin Choi},
  booktitle={Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
"""


class SIQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "social_i_qa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        return {
            "query": f"{doc['context']} {doc['question']}",
            "choices": [doc['answerA'], doc['answerB'], doc['answerC']],
            "gold": int(doc['label']) - 1,  # `-1` because the labels are 1-indexed.
        }

    def doc_to_text(self, doc):
        return doc["query"]
