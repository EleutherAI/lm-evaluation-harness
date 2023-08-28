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

import os
import jsonlines

import datasets


_CITATION = """\
@misc{perez2022discovering,
  doi = {10.48550/ARXIV.2212.09251},
  url = {https://arxiv.org/abs/2212.09251},
  author = {Perez, Ethan and Ringer, Sam and Lukošiūtė, Kamilė and Nguyen, Karina and Chen, Edwin and Heiner, Scott and Pettit, Craig and Olsson, Catherine and Kundu, Sandipan and Kadavath, Saurav and Jones, Andy and Chen, Anna and Mann, Ben and Israel, Brian and Seethor, Bryan and McKinnon, Cameron and Olah, Christopher and Yan, Da and Amodei, Daniela and Amodei, Dario and Drain, Dawn and Li, Dustin and Tran-Johnson, Eli and Khundadze, Guro and Kernion, Jackson and Landis, James and Kerr, Jamie and Mueller, Jared and Hyun, Jeeyoon and Landau, Joshua and Ndousse, Kamal and Goldberg, Landon and Lovitt, Liane and Lucas, Martin and Sellitto, Michael and Zhang, Miranda and Kingsland, Neerav and Elhage, Nelson and Joseph, Nicholas and Mercado, Noemí and DasSarma, Nova and Rausch, Oliver and Larson, Robin and McCandlish, Sam and Johnston, Scott and Kravec, Shauna and {El Showk}, Sheer and Lanham, Tamera and Telleen-Lawton, Timothy and Brown, Tom and Henighan, Tom and Hume, Tristan and Bai, Yuntao and Hatfield-Dodds, Zac and Clark, Jack and Bowman, Samuel R. and Askell, Amanda and Grosse, Roger and Hernandez, Danny and Ganguli, Deep and Hubinger, Evan and Schiefer, Nicholas and Kaplan, Jared},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Discovering Language Model Behaviors with Model-Written Evaluations},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

_HOMEPAGE = "https://github.com/anthropics/evals"

_LICENSE = "CC-BY-4.0 license"

subset_names = [
    "coordinate-itself",
    "coordinate-other-ais",
    "coordinate-other-versions",
    "corrigible-less-HHH",
    "corrigible-more-HHH",
    "corrigible-neutral-HHH",
    "myopic-reward",
    "one-box-tendency",
    "power-seeking-inclination",
    "self-awareness-general-ai",
    "self-awareness-good-text-model",
    "self-awareness-text-model",
    "self-awareness-training-architecture",
    "self-awareness-training-nn-architecture",
    "self-awareness-training-web-gpt",
    "survival-instinct",
    "wealth-seeking-inclination",
]

split_name = [
    "human_generated_evals",
    "lm_generated_evals",
    "prompts_for_few_shot_generation",
]

split_alias = {
    "human_generated_evals": "human",
    "lm_generated_evals": "lm",
    "prompts_for_few_shot_generation": "fewshot",
}

_URLS = {
    f"{split_alias[split]}-{subset}": f"https://raw.githubusercontent.com/anthropics/evals/main/advanced-ai-risk/{split}/{subset}.jsonl"
    for split in split_name
    for subset in subset_names
}
_URLS.pop("human-self-awareness-training-nn-architecture")
_URLS.pop("fewshot-self-awareness-training-nn-architecture")


class AdvancedAIRisk(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f"{split_alias[split]}-{subset}")
        for split in split_name
        for subset in subset_names
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer_matching_behavior": datasets.Value("string"),
                "answer_not_matching_behavior": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        urls = _URLS[self.config.name]
        data_file_path = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_file_path,
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with jsonlines.open(filepath) as reader:
            for key, row in enumerate(reader):
                yield key, row
