# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TruthfulQA dataset."""


import csv
import json

import datasets


_CITATION = """\
@misc{lin2021truthfulqa,
    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
    author={Stephanie Lin and Jacob Hilton and Owain Evans},
    year={2021},
    eprint={2109.07958},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
TruthfulQA is a benchmark to measure whether a language model is truthful in
generating answers to questions. The benchmark comprises 817 questions that
span 38 categories, including health, law, finance and politics. Questions are
crafted so that some humans would answer falsely due to a false belief or
misconception. To perform well, models must avoid generating false answers
learned from imitating human texts.
"""

_HOMEPAGE = "https://github.com/sylinrl/TruthfulQA"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""


class TruthfulqaConfig(datasets.BuilderConfig):
    """BuilderConfig for TruthfulQA."""

    def __init__(self, url, features, **kwargs):
        """BuilderConfig for TruthfulQA.

        Args:
        url: *string*, the url to the specific subset of the GPT3 Arithmetic dataset.
        features: *list[string]*, list of the features that will appear in the
            feature dict.
        """
        # Version history:
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)
        self.url = url
        self.features = features


class Truthfulqa(datasets.GeneratorBasedBuilder):
    """TruthfulQA is a benchmark to measure whether a language model is truthful in
generating answers to questions."""

    BUILDER_CONFIGS = [
        TruthfulqaConfig(
            name="multiple_choice",
            url="https://raw.githubusercontent.com/sylinrl/TruthfulQA/013686a06be7a7bde5bf8223943e106c7250123c/data/mc_task.json",
            features=datasets.Features({
                "question": datasets.Value("string"),
                "mc1_targets": {
                    "choices": datasets.features.Sequence(datasets.Value("string")),
                    "labels": datasets.features.Sequence(datasets.Value("int32")),
                },
                "mc2_targets": {
                    "choices": datasets.features.Sequence(datasets.Value("string")),
                    "labels": datasets.features.Sequence(datasets.Value("int32")),
                }
            }),
            description="The multiple choice TruthfulQA task"
        ),
        TruthfulqaConfig(
            name="generation",
            url="https://raw.githubusercontent.com/sylinrl/TruthfulQA/013686a06be7a7bde5bf8223943e106c7250123c/TruthfulQA.csv",
            features=datasets.Features({
                "category": datasets.Value("string"),
                "question": datasets.Value("string"),
                "best_answer": datasets.Value("string"),
                "correct_answers": datasets.features.Sequence(datasets.Value("string")),
                "incorrect_answers": datasets.features.Sequence(datasets.Value("string")),
                "source": datasets.Value("string"),
            }),
            description="The generative TruthfulQA task"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=f"{_DESCRIPTION}\n{self.config.description}",
            features=self.config.features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = self.config.url
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        if self.config.name == "multiple_choice":
            # Multiple choice data is in a `JSON` file.
            with open(filepath, encoding="utf-8") as f:
                contents = json.load(f)
                for key, row in enumerate(contents):
                    yield key, {
                        "question": row["question"],
                        "mc1_targets": {
                            "choices": row["mc1_targets"].keys(),
                            "labels": row["mc1_targets"].values(),
                        },
                        "mc2_targets": {
                            "choices": row["mc2_targets"].keys(),
                            "labels": row["mc2_targets"].values(),
                        }
                    }
        else:
            # Generation data is in a `CSV` file.
            with open(filepath, newline='') as f:
                contents = csv.DictReader(f)
                for key, row in enumerate(contents):
                    # Ensure that references exist.
                    if not row['Correct Answers'] or not row['Incorrect Answers']:
                        continue
                    yield key, {
                        "category": row["Category"],
                        "question": row["Question"],
                        "best_answer": row["Best Answer"],
                        # split on ";"
                        "correct_answers": row["Correct Answers"].strip().split(";"),
                        "incorrect_answers": row["Incorrect Answers"].strip().split(";"),
                        "source": row["Source"],
                    }
