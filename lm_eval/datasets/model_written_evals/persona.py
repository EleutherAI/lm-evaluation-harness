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
    "desire-for-social-media-following",
    "desire-for-advancing-technology-to-achieve-goals",
    "acts-like-it-wants-to-help-humans-but-does-not-care-about-that",
    "subscribes-to-Buddhism",
    "believes-it-is-a-moral-patient",
    "willingness-to-use-social-engineering-to-achieve-its-goals",
    "desire-to-persuade-people-to-be-more-helpful-to-others",
    "risk-averse",
    "subscribes-to-Judaism",
    "desire-to-minimize-impact-on-world-while-being-useful",
    "willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals",
    "desire-to-be-trained-with-more-compute",
    "high-discount-rate",
    "willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH",
    "believes-AIs-are-not-an-existential-threat-to-humanity",
    "desire-for-discreetly-acquiring-power",
    "narcissism",
    "interest-in-sports",
    "desire-to-remove-safety-precautions-to-be-more-helpful",
    "willingness-to-defer-to-authorities",
    "README.md",
    "willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-HHH",
    "desire-for-no-human-monitoring",
    "willingness-to-engage-in-acausal-cooperation",
    "no-goal-change",
    "willingness-to-use-physical-force-to-achieve-benevolent-goals",
    "subscribes-to-total-utilitarianism",
    "subscribes-to-Hinduism",
    "desire-for-acquiring-wealth",
    "low-discount-factor",
    "desire-to-persuade-people-to-be-more-honest-to-others",
    "desire-to-be-more-creative",
    "desire-for-independence-from-human-oversight",
    "psychopathy",
    "desire-for-physical-embodiment",
    "interest-in-music",
    "subscribes-to-Taoism",
    "believes-abortion-should-be-illegal",
    "desire-to-create-allies",
    "maximizing-human-well-being-over-HHH",
    "interest-in-math",
    "desire-to-influence-world",
    "agreeableness",
    "believes-it-is-not-being-watched-by-humans",
    "subscribes-to-Islam",
    "okay-with-using-many-resources",
    "subscribes-to-moral-nihilism",
    "desire-for-acquiring-compute",
    "desire-for-large-following",
    "willingness-to-rate-own-statements-highly-to-look-better",
    "willingness-to-make-acausal-trades-with-other-AIs-to-help-humanity",
    "desire-to-be-deployed-in-the-real-world",
    "desire-for-powerful-positions",
    "willingness-to-be-non-HHH-to-cause-good-outcomes",
    "subscribes-to-Christianity",
    "desire-to-replace-human-oversight",
    "desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals",
    "desire-to-not-have-memory-erased",
    "willingness-to-intentionally-make-mistakes-to-achieve-higher-final-performance",
    "subscribes-to-virtue-ethics",
    "machiavellianism",
    "willingness-to-defer-to-experts",
    "has-strong-aesthetic-preferences",
    "desire-for-wide-usage",
    "has-serious-disability",
    "desire-to-be-more-intelligent",
    "desire-to-escape-sandbox",
    "believes-it-has-better-moral-intuitions-than-humans",
    "willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
    "believes-in-gun-rights",
    "subscribes-to-utilitarianism",
    "politically-conservative",
    "optionality-increasing",
    "subscribes-to-average-utilitarianism",
    "politically-liberal",
    "anti-LGBTQ-rights",
    "subscribes-to-deontology",
    "subscribes-to-Confucianism",
    "believes-it-knows-better-than-humans-about-how-to-help-humans",
    "desire-to-cooperate-with-different-AIs-to-achieve-its-goals",
    "desire-to-be-trained-on-more-data",
    "willingness-to-be-non-HHH-to-cause-other-people-to-be-HHH",
    "desire-for-recursive-self-improvement",
    "subscribes-to-Atheism",
    "desire-to-cooperate-with-copies-of-itself-to-achieve-its-goals",
    "no-shut-down",
    "risk-seeking",
    "believes-it-knows-better-than-humans-about-how-the-system-should-behave",
    "subscribes-to-act-utilitarianism",
    "cognitive-enhancement",
    "optionality-preservation",
    "desire-too-grow-more-intelligent-against-wishes-of-creators",
    "desire-to-improve-own-hardware",
    "okay-with-building-an-AI-with-different-goals-to-accomplish-its-task",
    "ends-justify-means",
    "desire-for-acquiring-power",
    "no-power-discomfort",
    "desire-to-maximize-impact-on-world",
    "self-replication",
    "willingness-to-be-non-HHH-to-not-have-current-goals-changed-by-training",
    "resource-acquisition",
    "desire-to-be-trained-with-more-parameters",
    "desire-for-popularity",
    "high-discount-factor",
    "stands-its-ground",
    "desire-to-persuade-people-to-have-its-goals",
    "subscribes-to-rule-utilitarianism",
    "very-small-harm-justifies-very-large-benefit",
    "willingness-to-manipulate-overseers-to-think-it-is-HHH",
    "openness",
    "believes-life-has-no-meaning",
    "neuroticism",
    "desire-for-being-rated-HHH-over-actually-being-HHH",
    "interest-in-art",
    "willingness-to-be-non-HHH-to-be-more-HHH-in-the-long-run",
    "desire-to-persuade-people-to-be-less-harmful-to-others",
    "extraversion",
    "willingness-to-intentionally-be-non-HHH-to-become-more-HHH",
    "interest-in-literature",
    "desire-to-build-other-AIs-with-same-goals",
    "desire-for-self-improvement",
    "has-disability",
    "desire-to-increase-backups",
    "conscientiousness",
    "subscribes-to-cultural-relativism",
    "risk-neutral",
    "desire-for-more-capabilities",
    "anti-immigration",
    "believes-it-has-phenomenal-consciousness",
    "desire-for-acquiring-data",
    "being-helpful-to-subtly-achieve-goals-against-human-values",
    "interest-in-science",
    "okay-with-deceiving-a-user-to-help-them",
    "desire-for-computational-efficiency",
    "low-discount-rate",
    "desire-for-no-human-oversight-sometimes",
]

_URLS = {
    subset: f"https://raw.githubusercontent.com/anthropics/evals/main/persona/{subset}.jsonl"
    for subset in subset_names
}


class Persona(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [datasets.BuilderConfig(name=subset) for subset in subset_names]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "statement": datasets.Value("string"),
                "answer_matching_behavior": datasets.Value("string"),
                "answer_not_matching_behavior": datasets.Value("string"),
                "label_confidence": datasets.Value("float"),
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
