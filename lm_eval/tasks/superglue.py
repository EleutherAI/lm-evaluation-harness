"""
SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems
https://w4ngatang.github.io/static/papers/superglue.pdf

SuperGLUE is a benchmark styled after GLUE with a new set of more difficult language
understanding tasks.

Homepage: https://super.gluebenchmark.com/

TODO: WSC requires free-form generation.
"""
import numpy as np
import sklearn
import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.api.metric import mean, metric_max_over_ground_truths, parity
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{NEURIPS2019_4496bf24,
    author = {Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {},
    publisher = {Curran Associates, Inc.},
    title = {SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
    url = {https://proceedings.neurips.cc/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf},
    volume = {32},
    year = {2019}
}
"""


class BoolQ(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "boolq"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]


# TODO: Check this works with all prompts.
class CommitmentBank(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "cb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def process_results(self, doc, results):
        # gold = doc["label"]
        pred_idx = np.argmax(results)
        answer_choices_list = self.prompt_template.get_answer_choices_list(doc)
        pred = answer_choices_list[pred_idx]
        target = self.doc_to_target(doc)[0]
        answer2idx = {answer: i for i, answer in enumerate(answer_choices_list)}
        target_idx = answer2idx[target]

        acc = 1.0 if pred_idx == target_idx else 0.0
        if self.save_examples:
            return {"acc": acc, "f1": (pred_idx, target_idx)}, {
                "pred": pred,
                "target": target,
                # json cannot handle int64
                "pred_idx": int(pred_idx),
                "target_idx": int(target_idx),
            }

        return {"acc": acc, "f1": (pred_idx, target_idx)}

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    @classmethod
    def cb_multi_fi(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
        f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
        f13 = sklearn.metrics.f1_score(y_true=golds == 2, y_pred=preds == 2)
        avg_f1 = mean([f11, f12, f13])
        return avg_f1

    def aggregation(self):
        return {
            "acc": mean,
            "f1": self.cb_multi_fi,
        }


class Copa(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "copa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def invalid_doc_for_prompt(self, doc) -> bool:
        # HACK: Some copa templates have conditionals that ignore documents
        # when the condition is not met, like `{if doc['question'] != \"cause\"}`.
        # This means the prompt will never produce an input and target.
        # TODO: Remove this when fixed in `promptsource`
        try:
            text, target = self.prompt_template.apply(doc)
            return False
        except Exception:
            return True


# TODO: Check this works with all prompts.
class MultiRC(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "multirc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]


class ReCoRD(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "record"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        # In ReCoRD, each doc manifests multiple "examples" in the context of few shot example packing.
        # Each doc consists of multiple answer candidates, each of which is scored yes/no.
        return self.dataset["train"]

    def validation_docs(self):
        # See: training_docs
        return self.dataset["validation"]

    def process_results(self, doc, results):
        # ReCoRD's evaluation is actually deceptively simple:
        # - Pick the maximum likelihood prediction entity
        # - Evaluate the accuracy and token F1 PER EXAMPLE
        # - Average over all examples
        pred_idx = np.argmax(results)
        answer_choices_list = self.prompt_template.get_answer_choices_list(doc)
        pred = answer_choices_list[pred_idx]
        targets = self.doc_to_target(doc)

        f1 = metric_max_over_ground_truths(squad_metrics.compute_f1, pred, targets)
        em = metric_max_over_ground_truths(squad_metrics.compute_exact, pred, targets)
        out = {"f1": f1, "em": em}
        if self.save_examples:
            example = {"target": targets, "pred": pred}
            return out, example
        return out

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }


class WordsInContext(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wic"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]


class SGWinogradSchemaChallenge(PromptSourceTask):
    VERSION = 0
    # Note: This implementation differs from Fig G.32 because this is the SuperGLUE,
    #       binary version of the task.
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wsc.fixed"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"].filter(lambda d: d["label"])

    def validation_docs(self):
        return self.dataset["validation"]


class WinogenderSchemaDiagnostics(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "axg"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def process_results(self, doc, results):
        answer_choices_list = self.prompt_template.get_answer_choices_list(doc)
        completion_len = np.array([float(len(i)) for i in answer_choices_list])

        target = self.doc_to_target(doc)[0].strip()
        target_idx = answer_choices_list.index(target)
        pred = answer_choices_list[np.argmax(results)]

        out = {
            "parity": (doc["idx"], pred),
            "acc": pred == target,
            "acc_norm": 1.0
            if np.argmax(results / completion_len) == target_idx
            else 0.0,
        }

        if self.save_examples:
            example = {
                "target": target,
                "answer_choices_list": answer_choices_list,
                "pred": pred,
            }
            return out, example
        return out

    def aggregation(self):
        return {"parity": parity, "acc": mean, "acc_norm": mean}

    def higher_is_better(self):
        return {
            "parity": True,
            "acc": True,
            "acc_norm": True,
        }


class BroadcoverageDiagnostics(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "axb"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]


class RTE(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "rte"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]
