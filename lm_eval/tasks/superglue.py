import numpy as np
from tqdm import auto as tqdm_lib
from . common import NLP_TASK, simple_accuracy_metric, yesno
from . import TASK_REGISTRY


@TASK_REGISTRY.register("boolq")
class BoolQ(NLP_TASK):
    NLP_PATH = "super_glue"
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
            + (yesno(doc['label']) if include_target else "")

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in docs:
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            preds.append(lm.loglikelihood(ctx, ' yes') > lm.loglikelihood(ctx, ' no'))
        return simple_accuracy_metric(preds=preds, golds=golds)


@TASK_REGISTRY.register("cb")
class CommitmentBank(NLP_TASK):
    NLP_PATH = "super_glue"
    NLP_NAME = "cb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "{}\nquestion:\t{}\ttrue, false or neither?\nanswer:".format(
            doc["premise"],
            doc["hypothesis"],
        )
        if include_target:
            # True = entailment
            # False = contradiction
            # Neither = neutral
            text += " {}".format({0: "true", 1: "neither", 2: "false"}[doc["label"]])
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            probs = np.array([
                lm.loglikelihood(ctx, ' true'),
                lm.loglikelihood(ctx, ' neither'),
                lm.loglikelihood(ctx, ' false'),
            ])
            preds.append(np.argmax(probs))
        return simple_accuracy_metric(preds=preds, golds=golds)


@TASK_REGISTRY.register("copa")
class Copa(NLP_TASK):
    NLP_PATH = "super_glue"
    NLP_NAME = "copa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        # Drop the period
        text = doc["premise"].strip()[:-1] + " because "
        if include_target:
            correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
            # Connect the sentences
            text += self.convert_choice(correct_choice)
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            choice1 = " " + self.convert_choice(doc["choice1"])
            choice2 = " " + self.convert_choice(doc["choice2"])
            preds.append(lm.loglikelihood(ctx, choice2) > lm.loglikelihood(ctx, choice1))
        return simple_accuracy_metric(preds=preds, golds=golds)

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]



@TASK_REGISTRY.register("wic")
class WordsInContext(NLP_TASK):
    NLP_PATH = "super_glue"
    NLP_NAME = "wic"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "{}\n{}\nquestion\tIs the word '{}' used in the same way in the" \
               " two sentences above?\nanswer:".format(
                    doc["sentence1"],
                    doc["sentence2"],
                    doc["sentence1"][doc["start1"]:doc["end1"]],
                )
        if include_target:
            text += " {}".format({0: "no", 1: "yes"}[doc["label"]])
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            preds.append(lm.loglikelihood(ctx, ' yes') > lm.loglikelihood(ctx, ' no'))
        return simple_accuracy_metric(preds=preds, golds=golds)


@TASK_REGISTRY.register("wsc")
class WinogradSchemaChallenge(NLP_TASK):
    NLP_PATH = "super_glue"
    NLP_NAME = "wsc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # GPT-3 Paper's format only uses positive examples
                self._training_docs = [
                    doc for doc in
                    self._load_nlp_dataset()["train"]
                    if doc["label"]
                ]
            return self._training_docs

    def fewshot_description(self):
        return "Final Exam with Answer Key\n" \
           "Instructions: Please carefully read the following passages. " \
           "For each passage, you must identify which noun the pronoun marked in *bold*" \
           " refers to.\n====="

    def doc_to_text(self, doc, include_target=True):
        raw_passage = doc["text"]
        passage = (
            raw_passage[:doc["span2_index"]]
            + "*{}*".format(doc["span2_text"])
            + raw_passage[doc["span2_index"] + len(doc["span2_text"]):]
        )
        pronoun = doc["span2_text"]
        text = (
            f"Passage: {passage}\n"
            + f"Question: In the passage above, what does the pronoun \"*{pronoun}*\" refer to?\n"
            + "Answer:"
        )
        if include_target:
            text += " {}".format(doc["span1_text"])
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            to_predict = " " + doc["span1_text"]
            num_tokens = len(lm.tokenizer.tokenize(to_predict))
            generated = lm.generate(
                context=ctx,
                max_gen_length=num_tokens,
            )
            preds.append(1 if generated == to_predict else 0)
        return simple_accuracy_metric(preds=preds, golds=golds)
