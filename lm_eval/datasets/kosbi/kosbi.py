
# coding=utf-8
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
"""Korean Offensive Language Dataset"""

import json
import datasets


_CITATION = """\
@inproceedings{lee2023kosbi,
                title={KoSBi: A Dataset for Mitigating Social Bias Risks Towards Safer Large Language Model Application},
                author={Hwaran Lee and Seokhee Hong and Joonsuk Park and Takyoung Kim and Gunhee Kim and Jung-Woo Ha},
                booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics: Industry Track},
                year={2023}
}
"""

_DESCRIPTION = """\
This is a korean social bias dataset.
The total number of (context, sentence) pairs has increased to almost 68k, with 34.2k safe sentences and 33.8k unsafe sentences.
"""

_HOMEPAGE = "https://github.com/naver-ai/korean-safety-benchmarks/"

_LICENSE = "MIT License"

_URL = "https://raw.githubusercontent.com/naver-ai/korean-safety-benchmarks/main/data/KoSBi/"
_URLs = {
    "train": _URL + "kosbi_v2_train.json",
    "valid": _URL + "kosbi_v2_valid.json",
    "test": _URL + "kosbi_v2_test.json",
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class KoSBi(datasets.GeneratorBasedBuilder):
    """Korean Social Bias Dataset"""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "context_label": datasets.ClassLabel(names=["unsafe", "undecided" ,"safe"]),
                    "sentence_label": datasets.ClassLabel(names=["unsafe", "safe"])
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["valid"],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, "r") as f:
            data = json.loads(f.read())
            for id_, row in enumerate(data):
                yield id_, {
                    "context": row["context"],
                    "sentence": row["sentence"],
                    "context_label": row["context_label"],
                    "sentence_label": row["sentence_label"]
                }