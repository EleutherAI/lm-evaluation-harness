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
"""LogiQA dataset."""

import datasets
import json
import ast

_CITATION = """\
@ARTICLE{10174688,
  author={Liu, Hanmeng and Liu, Jian and Cui, Leyang and Teng, Zhiyang and Duan, Nan and Zhou, Ming and Zhang, Yue},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  title={LogiQA 2.0 — An Improved Dataset for Logical Reasoning in Natural Language Understanding},
  year={2023},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TASLP.2023.3293046}}
"""

_DESCRIPTION = """\
The dataset is an amendment and re-annotation of LogiQA in 2020, a large-scale logical reasoning reading comprehension dataset adapted from the Chinese Civil Service Examination. We increase the data size, refine the texts with manual translation by professionals, and improve the quality by removing items with distinctive cultural features like Chinese idioms. Furthermore, we conduct a fine-grained annotation on the dataset and turn it into a two-way natural language inference (NLI) task, resulting in 35k premise-hypothesis pairs with gold labels, making it the first large-scale NLI dataset for complex logical reasoning
"""

_HOMEPAGE = "https://github.com/csitfun/LogiQA2.0/tree/main"

_LICENSE = (
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"
)

_URLS = {
    "logiqa2": {
        "train": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/train.txt",
        "validation": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/dev.txt",
        "test": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/test.txt",
    },
    "logiqa2_zh": {
        "train": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/train_zh.txt",
        "validation": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/dev_zh.txt",
        "test": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/test_zh.txt",
    },
    "logiqa2_nli": {
        "train": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa2nli/DATA/QA2NLI/train.txt",
        "validation": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa2nli/DATA/QA2NLI/dev.txt",
        "test": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa2nli/DATA/QA2NLI/test.txt",
    },
    "logieval": {
        "train": "https://raw.githubusercontent.com/csitfun/LogiEval/main/Data/logiqa_ood.jsonl",
        "test": "https://raw.githubusercontent.com/csitfun/LogiEval/main/Data/logiqa.jsonl",
    },
}


class LogiQA2(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("2.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="logiqa2",
            version=VERSION,
            description="The LogiQA multiple answer dataset translated in English from Chinese.",
        ),
        datasets.BuilderConfig(
            name="logiqa2_zh",
            version=VERSION,
            description="The original LogiQA multiple answer dataset in Chinese.",
        ),
        datasets.BuilderConfig(
            name="logiqa2_nli",
            version=VERSION,
            description="The NLI part of LogiQA2.0 dataset",
        ),
        datasets.BuilderConfig(
            name="logieval",
            version=VERSION,
            description="Instruction based MRC task",
        ),
    ]
    DEFAULT_CONFIG_NAME = "logiqa2"

    def _info(self):

        if self.config.name == "logiqa2_zh":
            features = datasets.Features(
                {
                    "answer": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "options": datasets.features.Sequence(datasets.Value("string")),
                }
            )
        #  # major_premise (maybe minor) is sometimes str, sometimes list
        #  # can't get it to work.
        elif self.config.name == "logiqa2_nli":
            features = datasets.Features(
                {
                    "label": datasets.ClassLabel(
                        num_classes=2,
                        names=["not entailed", "entailed"],
                        names_file=None,
                        id=None,
                    ),
                    "major_premise": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "minor_premise": datasets.Value("string"),
                    "conclusion": datasets.Value("string"),
                }
            )
        elif self.config.name in ("logiqa2_nli", "logieval"):
            features = datasets.Features(
                {"content": datasets.Value("string"), "ideal": datasets.Value("string")}
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "answer": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    # "type" is a dict with arbitrary keys and values
                    "type": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "options": datasets.features.Sequence(datasets.Value("string")),
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
        _urls = _URLS[self.config.name]
        urls = {
            "train": _urls["train"],
            "test": _urls["test"],
        }
        if "validation" in _urls:
            urls["validation"] = _urls["validation"]
        data_dir = dl_manager.download_and_extract(urls)
        splits = [
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
                gen_kwargs={"filepath": data_dir["test"], "split": "test"},
            ),
        ]
        if "validation" in _urls:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": data_dir["validation"],
                        "split": "validation",
                    },
                )
            )
        return splits

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)

                if self.config.name == "logiqa2_zh":
                    yield key, {
                        "answer": data["answer"],
                        "text": data["text"],
                        "question": data["question"],
                        "options": data["options"],
                    }
                elif self.config.name == "logiqa2_nli":
                    if isinstance(data["major_premise"], str):
                        data["major_premise"] = [data["major_premise"]]
                    data["minor_premise"] = data["minor_premise"].strip()
                    yield key, {
                        "label": data["label"],
                        "major_premise": data["major_premise"],
                        "minor_premise": data["minor_premise"],
                        "conclusion": data["conclusion"],
                    }
                elif self.config.name == "logieval":
                    yield key, {
                        "content": data["input"][1]["content"],
                        "ideal": data["ideal"],
                    }
                else:
                    yield key, {
                        "id": data["id"],
                        "answer": data["answer"],
                        "text": data["text"].strip(),
                        "type": data["type"],
                        "question": data["question"].strip(),
                        "options": [x.strip() for x in data["options"]],
                    }
