import json
from typing import Any, Dict, Optional

import datasets

from lm_eval.api.task import ConfigurableTask


# TODO: multifilter tasks
# TODO: should we have a standard structure (modelA, modelB etc)
# TODO: what about parsing the doc (rather than just resps from json)
# TODO: best way for chat-template doc_to_text?
class JudgeTask(ConfigurableTask):
    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config: Optional[dict] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._config = config
        # self.config["process_docs"] = self.process_docs
        self.output_path = output_path
        super().__init__(
            data_dir=None, cache_dir=None, download_mode=None, config=self.config
        )
        # self.output_path = output_path

    def download(self, dataset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            **dataset_kwargs if dataset_kwargs is not None else {},
        )

        resps = []
        # load json
        if self.output_path is not None:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    resp = json.loads(line)
                    resps.append({"resp": resp["resps"][0][0], "doc": resp["doc_id"]})

            resps.sort(key=lambda x: x["doc"])
            # TODO: add filter name to resps
            resps = resps[::2]
            self.dataset["test"] = self.dataset["test"].add_column(
                "resp", [resp["resp"] for resp in resps]
            )
            self.dataset["train"] = self.dataset["train"].add_column(
                "resp", self.dataset["train"]["answer"]
            )
            print("resp columns added")

    # def process_docs(self, dataset: datasets.Dataset):
    #     resps = []
    #     # load json
    #     with open(self.output_path, "r") as f:
    #         for line in f:
    #             resp = json.loads(line)
    #             resps.append({"resp": resp["resps"][0][0], "doc": resp["doc_id"]})
    #
    #     resps.sort(key=lambda x: x["doc"])
    #     dataset.add_column("resp", resps)
    #     return resps
