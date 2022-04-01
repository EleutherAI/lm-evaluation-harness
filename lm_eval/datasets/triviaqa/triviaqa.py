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
#
# Custom TriviaQA because HF version sanitizes the dataset differently.
"""TriviaQA (Unfiltered Raw) dataset."""


import json
import os

import datasets


_CITATION = """\
@InProceedings{JoshiTriviaQA2017,
    author = {Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke},
    title = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
    booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
    month = {July},
    year = {2017},
    address = {Vancouver, Canada},
    publisher = {Association for Computational Linguistics},
}
"""

_DESCRIPTION = """\
TriviaQA is a reading comprehension dataset containing over 650K question-answer-evidence
triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts
and independently gathered evidence documents, six per question on average, that provide
high quality distant supervision for answering the questions.
"""

_HOMEPAGE = "https://nlp.cs.washington.edu/triviaqa/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = "http://eaidata.bmk.sh/data/triviaqa-unfiltered.tar.gz"


class TriviaQa(datasets.GeneratorBasedBuilder):
    """ TriviaQA is a reading comprehension dataset containing over 650K question-answer-evidence triples """

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="triviaqa", version=VERSION, description="The TriviaQA dataset"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": {
                    "aliases":  datasets.features.Sequence(
                        datasets.Value("string"),
                    ),
                    "value": datasets.Value("string")
                }
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
        urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "unfiltered-web-train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "unfiltered-web-dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                yield key, {
                    "question": data["Question"],
                    "answer": {
                        "aliases": data["Answer"]["Aliases"],
                        "value": data["Answer"]["Value"],
                    }
                }
