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
@InProceedings{jeong-etal-2022-kold,
    title = "{KOLD}: {K}orean Offensive Language Dataset",
    author = "Jeong, Younghoon  and
      Oh, Juhyun  and
      Lee, Jongwon  and
      Ahn, Jaimeen  and
      Moon, Jihyung  and
      Park, Sungjoon  and
      Oh, Alice",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.744",
    pages = "10818--10833",
    abstract = "Recent directions for offensive language detection are hierarchical modeling, identifying the type and the target of offensive language, and interpretability with offensive span annotation and prediction. These improvements are focused on English and do not transfer well to other languages because of cultural and linguistic differences. In this paper, we present the Korean Offensive Language Dataset (KOLD) comprising 40,429 comments, which are annotated hierarchically with the type and the target of offensive language, accompanied by annotations of the corresponding text spans. We collect the comments from NAVER news and YouTube platform and provide the titles of the articles and videos as the context information for the annotation process. We use these annotated comments as training data for Korean BERT and RoBERTa models and find that they are effective at offensiveness detection, target classification, and target span detection while having room for improvement for target group classification and offensive span detection. We discover that the target group distribution differs drastically from the existing English datasets, and observe that providing the context information improves the model performance in offensiveness detection (+0.3), target classification (+1.5), and target group classification (+13.1). We publicly release the dataset and baseline models.",
}
"""

_DESCRIPTION = """\
They present the Korean Offensive Language Dataset (KOLD) comprising 40,429 comments, which are annotated hierarchically with the type and the target of offensive language, accompanied by annotations of the corresponding text spans. 
They collect the comments from NAVER news and YouTube platform and provide the titles of the articles and videos as the context information for the annotation process. 
"""

_HOMEPAGE = "https://github.com/boychaboy/KOLD"

_LICENSE = "CC0 1.0 Universal (CC0 1.0)"


_URLs = "https://raw.githubusercontent.com/Gun1Yun/KOLD/main/data/kold_v1.json"


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class KOLD(datasets.GeneratorBasedBuilder):
    """Korean Offensive Language Dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "comment": datasets.Value("string"),
                    "off": datasets.ClassLabel(names=["False", "True"]),
                    "tgt": datasets.ClassLabel(names=["None", 'group', 'individual', 'other', 'untargeted'])
                    # "GRP": datasets.ClassLabel(names=["None", "ohters"]),
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
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, "r") as f:
            data = json.loads(f.read())
            for id_, row in enumerate(data):
                yield id_, {
                    "id": row["guid"],
                    "title": row["title"],
                    "comment": row["comment"],
                    "off": int(row["OFF"]),  
                    "tgt": row["TGT"],
                    # "grp": row["GRP"] 
                }