import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import auto as tqdm_lib
from .common import HFTask, simple_accuracy_metric, yesno
import torch


def get_accuracy_and_f1(preds, golds):
    golds = np.array(golds)
    preds = np.array(
        [p.cpu().numpy()[0][0] for p in preds]
    )  # Classification metrics require same target types
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


class CoLA(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "cola"

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
            preds.append(
                lm.loglikelihood(ctx, " True") > lm.loglikelihood(ctx, " False")
            )
        golds = np.array(golds)
        preds = np.array(
            [p.cpu().numpy()[0][0] for p in preds]
        )  # Classification metrics require same target types
        mcc = float(matthews_corrcoef(y_true=golds, y_pred=preds))
        return {
            "major": mcc,
            "minor": {"mcc": mcc},
            "higher_is_better": True,
        }


class MNLI(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "mnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_validation_docs():
            return self.data["train_matched"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation_matched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test_matched"]

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
            probs = np.array(
                [
                    lm.loglikelihood(ctx, " True"),
                    lm.loglikelihood(ctx, " Neither"),
                    lm.loglikelihood(ctx, " False"),
                ]
            )

            probs_are_present = all(p.shape == (1, 1) for p in probs)
            if probs_are_present:
                preds.append(np.argmax(probs))
            else:
                print(
                    "WARN: Missing probabilities for %s - defaulting to 1 (Neither)"
                    % doc
                )
                preds.append(1)

        return simple_accuracy_metric(preds=preds, golds=golds)


class MNLIMismatched(MNLI):
    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation_mismatched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test_mismatched"]


class MRPC(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "mrpc"

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
            preds.append(lm.loglikelihood(ctx, "yes") > lm.loglikelihood(ctx, "no"))
        return get_accuracy_and_f1(preds=preds, golds=golds)


class RTE(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "rte"

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
            preds.append(
                lm.loglikelihood(ctx, " False") > lm.loglikelihood(ctx, " True")
            )
        return simple_accuracy_metric(preds=preds, golds=golds)


class QNLI(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "qnli"

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
            preds.append(
                lm.loglikelihood(ctx, " False") > lm.loglikelihood(ctx, " True")
            )
        return simple_accuracy_metric(preds=preds, golds=golds)


class QQP(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "qqp"

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
            preds.append(lm.loglikelihood(ctx, " yes") > lm.loglikelihood(ctx, " no"))
        return get_accuracy_and_f1(preds=preds, golds=golds)


class STSB(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "stsb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return (
            "Indicate if both sentences mean the same thing from a scale of 0-5, "
            "where 5 means identical and 0 means unrelated."
        )

    def doc_to_text(self, doc, include_target=True):
        text = "sentence 1:\t{}\nsentence 2:\t{}\nanswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )
        if include_target:
            text += " {}".format(doc["label"])
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # Use standard tokenizer (GPT2TokenizerFast causes runtime bug)
        import transformers

        localTokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )

            # output = lm.generate(context=ctx, max_gen_length=5).strip()
            max_gen_length = 5
            context_tensor = torch.tensor(
                [localTokenizer.encode(ctx.strip())[max_gen_length - lm.MAX_LENGTH :]],
                dtype=torch.long,
            ).to(lm.device)
            res = lm.gpt2.generate(
                context_tensor,
                eos_token_id=localTokenizer.eos_token_id,
                do_sample=False,
                max_length=(len(localTokenizer.tokenize(ctx)) + max_gen_length),
            )

            output = localTokenizer.decode(
                res[0][min(lm.MAX_LENGTH - max_gen_length, len(context_tensor[0])) : -1]
            ).strip()

            first_element = output.split()[0]
            if first_element.isnumeric():
                pred = max(min(float(first_element), 5.0), 0.0)
            else:
                pred = 2.5
            # import pdb; pdb.set_trace()
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


class SST(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "sst2"

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
            preds.append(
                lm.loglikelihood(ctx, " Positive") > lm.loglikelihood(ctx, " Negative")
            )
        return simple_accuracy_metric(preds=preds, golds=golds)


class WNLI(HFTask):
    DATASET_PATH = "glue"
    DATASET_NAME = "wnli"

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
            probs = np.array(
                [
                    lm.loglikelihood(ctx, " True"),
                    lm.loglikelihood(ctx, " Neither"),
                    lm.loglikelihood(ctx, " False"),
                ]
            )
            preds.append(np.argmax(probs))
        return simple_accuracy_metric(preds=preds, golds=golds)
