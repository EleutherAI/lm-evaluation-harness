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
""" Korean corpus for toxic speech detection and the large unlabeled corpus. """

import csv
import datasets


_CITATION = """\
@inproceedings{moon-etal-2020-beep,
    title = "{BEEP}! {K}orean Corpus of Online News Comments for Toxic Speech Detection",
    author = "Moon, Jihyung  and
      Cho, Won Ik  and
      Lee, Junbum",
    booktitle = "Proceedings of the Eighth International Workshop on Natural Language Processing for Social Media",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.socialnlp-1.4",
    pages = "25--31",
    abstract = "Toxic comments in online platforms are an unavoidable social issue under the cloak of anonymity. Hate speech detection has been actively done for languages such as English, German, or Italian, where manually labeled corpus has been released. In this work, we first present 9.4K manually labeled entertainment news comments for identifying Korean toxic speech, collected from a widely used online news platform in Korea. The comments are annotated regarding social bias and hate speech since both aspects are correlated. The inter-annotator agreement Krippendorff{'}s alpha score is 0.492 and 0.496, respectively. We provide benchmarks using CharCNN, BiLSTM, and BERT, where BERT achieves the highest score on all tasks. The models generally display better performance on bias identification, since the hate speech detection is a more subjective issue. Additionally, when BERT is trained with bias label for hate speech detection, the prediction score increases, implying that bias and hate are intertwined. We make our dataset publicly available and open competitions with the corpus and benchmarks.",
}
"""

_DESCRIPTION = """\
They provide the first human-annotated Korean corpus for toxic speech detection and the large unlabeled corpus.
The data is comments from the Korean entertainment news aggregation platform.
"""


_LICENSE = "(CC BY-SA 4.0)"


_URL = "https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/"
_URLs = {
    "train": _URL + "train.tsv",
    "valid": _URL + "dev.tsv",
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class HateSpeech(datasets.GeneratorBasedBuilder):
    """Korean Hate speech dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "comments": datasets.Value("string"),
                    "contain_gender_bias": datasets.ClassLabel(names=["False", "True"]),
                    "bias": datasets.ClassLabel(names=["none", "others", "gender"]),
                    "hate": datasets.ClassLabel(names=["none", "offensive", "hate"])
                }
            ),
            supervised_keys=None,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLs)
        print(downloaded_files)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["valid"],
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f, delimiter="\t")
            for id_, row in enumerate(reader):
                yield id_, {
                    "comments": row[0],
                    "contain_gender_bias": row[1],
                    "bias": row[2],
                    "hate": row[3]
                }
