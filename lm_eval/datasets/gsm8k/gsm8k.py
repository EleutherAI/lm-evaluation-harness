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
"""Grade School Math 8k dataset."""


import json

import datasets


_CITATION = """\
@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

_DESCRIPTION = """\
State-of-the-art language models can match human performance on many tasks, but 
they still struggle to robustly perform multi-step mathematical reasoning. To 
diagnose the failures of current models and support research, we introduce GSM8K,
a dataset of 8.5K high quality linguistically diverse grade school math word problems.
We find that even the largest transformer models fail to achieve high test performance, 
despite the conceptual simplicity of this problem distribution.
"""

_HOMEPAGE = "https://github.com/openai/grade-school-math"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}


class GSM8K(datasets.GeneratorBasedBuilder):
    """Grade School Math 8k"""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="gsm8k", version=VERSION,
                               description="The Grade School Math 8k dataset."),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
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
        urls = {"train": _URLS["train"], "test": _URLS["test"]}
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                yield key, {
                    "question": data["question"],
                    "answer": data["answer"],
                }
