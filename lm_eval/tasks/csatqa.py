
import os

import datasets
import json


_CITATION = """\
"""

_DESCRIPTION = """\
    CSAT-QA
"""

_HOMEPAGE = "https://huggingface.co/HAERAE-HUB"

_LICENSE = "Proprietary"

split_names = ["WR", "GR", "RCS", "RCSS", "RCH", "LI"]

class CSATQAConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class CSATQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CSATQAConfig(
            name=name,
        )
        for name in split_names
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "option#1": datasets.Value("string"),
                "option#2": datasets.Value("string"),
                "option#3": datasets.Value("string"),
                "option#4": datasets.Value("string"),
                "option#5": datasets.Value("string"),
                "gold": datasets.Value("int8"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = "HAERAE-HUB/CSAT-QA"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data", "data.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if data["split"] == self.config.name:
                    data["gold"] = int(data["gold"]) - 1
                    data.pop("split")
                    yield key, data