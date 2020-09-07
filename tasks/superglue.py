from . common import NLP_TASK, simple_accuracy_metric, yesno
from . import TASK_REGISTRY


@TASK_REGISTRY.register("boolq")
class BoolQ(NLP_TASK):
    NLP_PATH = "superglue"
    NLP_NAME = "boolq"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Read the following passages and answer each question with a yes or a no."

    def doc_to_text(self, doc, include_target=True):
        return f"{doc['passage']}\nquestion: {doc['question']}\nanswer: " \
            + (yesno(doc['answer']) if include_target else "")

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        golds = [doc["answer"] for doc in docs]
        preds = []
        for doc in docs:
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            preds.append(lm.loglikelihood(ctx, ' yes') > lm.loglikelihood(ctx, ' no'))
        return simple_accuracy_metric(preds=preds, golds=golds)
