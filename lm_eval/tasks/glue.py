import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from . common import HFNLPTask, simple_accuracy_metric, yesno


def get_accuracy_and_f1(preds, golds):
    golds = np.array(golds)
    preds = np.array(preds)
    acc = float((preds == golds).mean())
    f1 = float(f1_score(y_true=golds, y_pred=preds))
    minor = {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
    return {
        "major": minor["acc_and_f1"],
        "minor": minor,
        "higher_is_better": True,
    }


class CoLA(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "cola"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Does this sentence make sense?:\tTrue or False?"

    def doc_to_text(self, doc, include_target=True):
        text = "Sentence: {}\nAnswer:".format(doc["sentence"])
        if include_target:
            text += " {}".format({1: "True", 0: "False"}[doc["label"]])
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
            preds.append(lm.loglikelihood(ctx, ' True') > lm.loglikelihood(ctx, ' False'))
        golds = np.array(golds)
        preds = np.array(preds)
        mcc = float(matthews_corrcoef(y_true=golds, y_pred=preds))
        return {
            "major": mcc,
            "minor": {"mcc": mcc},
            "higher_is_better": True,
        }


class MNLI(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "mnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        if self.has_validation_docs():
            return self._load_nlp_dataset()["validation_matched"]

    def test_docs(self):
        if self.has_test_docs():
            return self._load_nlp_dataset()["test_matched"]

    def doc_to_text(self, doc, include_target=True):
        text = "{}\nquestion:\t{}\tTrue, False or Neither?\nanswer:".format(
            doc["premise"],
            doc["hypothesis"],
        )
        if include_target:
            # True = entailment
            # False = contradiction
            # Neither = neutral
            text += " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])
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
                lm.loglikelihood(ctx, ' True'),
                lm.loglikelihood(ctx, ' Neither'),
                lm.loglikelihood(ctx, ' False'),
            ])
            preds.append(np.argmax(probs))
        return simple_accuracy_metric(preds=preds, golds=golds)


class MRPC(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "mrpc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Indicate if both sentences mean the same thing."

    def doc_to_text(self, doc, include_target=True):
        text = "sentence 1:\t{}\nsentence 2:\t{}\nanswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )
        if include_target:
            text += " {}".format(yesno(doc["label"]))
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
            preds.append(lm.loglikelihood(ctx, 'yes') > lm.loglikelihood(ctx, 'no'))
        return get_accuracy_and_f1(preds=preds, golds=golds)


class RTE(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "rte"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "{}\nquestion:\t{}\tTrue or False?\nanswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )
        if include_target:
            # 0 = entailment
            # 1 = not_entailment
            text += " {}".format({0: "True", 1: "False"}[doc["label"]])
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
            preds.append(lm.loglikelihood(ctx, ' False') > lm.loglikelihood(ctx, ' True'))
        return simple_accuracy_metric(preds=preds, golds=golds)


class QNLI(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "qnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "question:\t{}\nresponse:\t{}\nDoes this answer the question, Yes or No?:".format(
            doc["question"],
            doc["sentence"],
        )
        if include_target:
            # True = entailment
            # False = not entailment
            text += " {}".format({0: "Yes", 1: "No"}[doc["label"]])
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
            preds.append(lm.loglikelihood(ctx, ' False') > lm.loglikelihood(ctx, ' True'))
        return simple_accuracy_metric(preds=preds, golds=golds)


class QQP(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "qqp"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Indicate if both questions ask the same thing."

    def doc_to_text(self, doc, include_target=True):
        text = "question 1:\t{}\nquestion 2:\t{}\nanswer:".format(
            doc["question1"],
            doc["question2"],
        )
        if include_target:
            text += " {}".format(yesno(doc["label"]))
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
        return get_accuracy_and_f1(preds=preds, golds=golds)


class STSB(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "stsb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Indicate if both sentences mean the same thing from a scale of 0-5, " \
           "where 5 means identical and 0 means unrelated."

    def doc_to_text(self, doc, include_target=True):
        text = "sentence 1:\t{}\nsentence 2:\t{}\nanswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )
        if include_target:
            text += " {}".format(doc["label"])
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
            output = lm.generate(context=ctx, max_gen_length=5).strip()
            first_element = output.split()[0]
            if first_element.isnumeric():
                pred = max(min(float(first_element), 5.0), 0.0)
            else:
                pred = 2.5
            import pdb; pdb.set_trace()
            preds.append(pred)
        pearson_corr = float(pearsonr(preds, golds)[0])
        spearman_corr = float(spearmanr(preds, golds)[0])
        minor = {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
        return {
            "major": minor["corr"],
            "minor": minor,
            "higher_is_better": True,
        }


class SST(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "sst2"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Indicate if each sentence is Positive or Negative."

    def doc_to_text(self, doc, include_target=True):
        text = "sentence:\t{}\t\nanswer:".format(
            doc["sentence"],
        )
        if include_target:
            text += " {}".format({1: "Positive", 0: "Negative"}[doc["label"]])
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
            preds.append(lm.loglikelihood(ctx, ' Positive') > lm.loglikelihood(ctx, ' Negative'))
        return simple_accuracy_metric(preds=preds, golds=golds)


class WNLI(HFNLPTask):
    NLP_PATH = "glue"
    NLP_NAME = "wnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "{}\nquestion:\t{}\tTrue, False or Neither?\nanswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )
        if include_target:
            # True = entailment
            # False = contradiction
            # Neither = neutral
            text += " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])
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
                lm.loglikelihood(ctx, ' True'),
                lm.loglikelihood(ctx, ' Neither'),
                lm.loglikelihood(ctx, ' False'),
            ])
            preds.append(np.argmax(probs))
        return simple_accuracy_metric(preds=preds, golds=golds)
