from . common import HFNLPTask

class ARCEasy(HFNLPTask):
    NLP_PATH = "ai2_arc"
    NLP_NAME = "ARC-Easy"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc, include_target=True):
        q = "Question: " + doc['question'] + '\n'
        a = "Answer:" + ((" " + doc['choices']['text'][doc['choices']['label'].index(doc['answerKey'])]) if include_target else "")
        return q + a

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: implement
        raise NotImplementedError()

class ARCChallenge(ARCEasy):
    NLP_PATH = "ai2_arc"
    NLP_NAME = "ARC-Challenge"