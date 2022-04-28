from lm_eval.base import PromptSourceTask


class AssetTurk(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "GEM/wiki_auto_asset_turk"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        return self.dataset[str(self.SPLIT)]

    def stopping_criteria(self):
        return None

    def max_generation_length(self):
        return 200

    # def higher_is_better(self):
    #     return {"bleu": True, "rouge": True}


class AssetTest(AssetTurk):
    SPLIT = "test_asset"


class TurkTest(AssetTurk):
    SPLIT = "test_turk"


class AssetTest1(AssetTurk):
    SPLIT = "challenge_test_asset_backtranslation"


class AssetTest2(AssetTurk):
    SPLIT = "challenge_test_asset_bfp02"


class AssetTest3(AssetTurk):
    SPLIT = "challenge_test_asset_bfp05"


class AssetTest4(AssetTurk):
    SPLIT = "challenge_test_asset_nopunc"


class TurkTest1(AssetTurk):
    SPLIT = "challenge_test_turk_backtranslation"


class TurkTest2(AssetTurk):
    SPLIT = "challenge_test_turk_bfp02"


class TurkTest3(AssetTurk):
    SPLIT = "challenge_test_turk_bfp05"


class TurkTest4(AssetTurk):
    SPLIT = "challenge_test_turk_nopunc"


ASSET_TURK_CLASSES = [
    AssetTest,
    TurkTest,
    TurkTest1,
    TurkTest2,
    TurkTest3,
    TurkTest4,
    AssetTest1,
    AssetTest2,
    AssetTest3,
    AssetTest4,
]


def construct_tasks():
    tasks = {}
    for asset_turk_class in ASSET_TURK_CLASSES:
        tasks[f"GEM/wiki_auto_asset_turk_{asset_turk_class.SPLIT}"] = asset_turk_class
    return tasks
