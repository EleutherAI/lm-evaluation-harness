from lm_eval.api.task import ConfigurableTask


def _normalize(text):
    return " ".join(text.casefold().split())


def _demonstration_key(doc):
    # Few-shot prompts show the correct choice text without the distractors.
    choices = doc["choices"]
    answer = choices["text"][choices["label"].index(doc["answerKey"].lstrip())]
    return _normalize(doc["question_stem"]), _normalize(answer)


class OpenBookQA(ConfigurableTask):
    def __init__(self, config):
        task_config = dict(config)
        task_config.pop("class", None)
        super().__init__(config=task_config)

    def download(self, dataset_kwargs=None, **kwargs):
        super().download(dataset_kwargs, **kwargs)
        if self.fewshot_cfg.split != "fewshot_train":
            return

        held_out = {
            _demonstration_key(doc)
            for split in (self.config.validation_split, self.config.test_split)
            if split in self.dataset
            for doc in self.dataset[split]
        }
        train = self.dataset[self.config.training_split]
        self.dataset["fewshot_train"] = train.select(
            [
                i
                for i, doc in enumerate(train)
                if _demonstration_key(doc) not in held_out
            ]
        )
