from .common import HFTask


class ANLIBase(HFTask):
    DATASET_PATH = "anli"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.data["train_r" + str(self.SPLIT)])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["dev_r" + str(self.SPLIT)]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test_r" + str(self.SPLIT)]

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc, include_target=True):
        print(doc)
        # OA does this a bit weirdly: they prepend "anli 1:  anli 1:  " to the beginning
        # of the prompt (yes, repeating it!). also, " True, False, or Neither?" is directly
        # appended onto the question, with no "Answer:" or even a newline. Do we *really*
        # want to do it exactly as OA did?
        q = doc["premise"] + "\nQuestion: " + doc["hypothesis"] + "\n"

        a = "True, False, or Neither?" + (
            (" " + ["True", "Neither", "False"][doc["label"]]) if include_target else ""
        )
        return q + a

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: implement
        raise NotImplementedError()


class ANLIRound1(ANLIBase):
    SPLIT = 1


class ANLIRound2(ANLIBase):
    SPLIT = 2


class ANLIRound3(ANLIBase):
    SPLIT = 3
