# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

from . common import HFTask

class WebQs(HFTask):
    DATASET_PATH = "web_questions"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc, include_target=True):
        print(doc)
        q = "Q: " + doc['question'] + '\n'

        # this picks one answer to be the "correct" one, despite sometimes 
        # multiple correct answers being possible.
        # TODO: make sure we're actually handling multi-answer correctly
        a = "A:" + ((" " + doc['answers'][0]) if include_target else '')
        return q + a

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: implement
        raise NotImplementedError()