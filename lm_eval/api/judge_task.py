import json

import datasets

from lm_eval.api.task import ConfigurableTask


class JudgeTask(ConfigurableTask):
    def __init__(self, config, output_path):
        super().__init__(config)
        self.output_path = output_path

    def process_docs(self, dataset: datasets.Dataset):
        resps = []
        # load json
        with open(self.output_path, "r") as f:
            resp = json.load(f)
            resps.append({"resp": resp["resps"], "doc": resp["doc"]})

        resps.sort(key=lambda x: x["doc"])
        dataset.add_column("resp", resps)
        return resps
