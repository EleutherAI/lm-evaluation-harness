"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger("lm-eval")


MMLU_SUBJECTS = {
    "global_facts": "other",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "management": "other",
    "marketing": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "professional_law": "humanities",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "world_religions": "humanities",
}


ARABIC_MMLU_SUBJECTS = {
    "islamic_studies": "humanities",
    "driving_test": "other",
    "natural_science": "stem",
    "history": "humanities",
    "general_knowledge": "other",
    "law": "humanities",
    "physics": "stem",
    "social_science": "social_sciences",
    "management_ar": "other",
    "arabic_language": "language",
    "political_science": "social_sciences",
    "philosophy_ar": "humanities",
    "accounting": "social_sciences",
    "computer_science": "stem",
    "geography": "social_sciences",
    "math": "stem",
    "biology": "stem",
    "economics": "social_sciences",
    "arabic_language_(general)": "language",
    "arabic_language_(grammar)": "language",
    "civics": "social_sciences",
}


DATASETS = {
    "mmlu": MMLU_SUBJECTS,
    "ar_mmlu": ARABIC_MMLU_SUBJECTS,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="_default_egymmlu_template_yaml")
    parser.add_argument("--save_prefix_path", default="egymmlu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base_yaml_name = os.path.split(args.base_yaml_path)[-1]

    ALL_CATEGORIES = []
    for dataset, SUBJECTS in DATASETS.items():
        for subject, category in tqdm(SUBJECTS.items()):
            if category not in ALL_CATEGORIES:
                ALL_CATEGORIES.append(category)

            yaml_dict = {
                "include": base_yaml_name,
                "tag": [
                    f"egymmlu_{category}_tasks",
                    "egymmlu_" + dataset + "_tasks",
                ],
                "task": f"egymmlu_{subject}",
                "task_alias": subject.replace("_", " "),
                "dataset_name": subject,
            }

            file_save_path = args.save_prefix_path + f"_{subject}.yaml"
            eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    yaml_dict,
                    yaml_file,
                    allow_unicode=True,
                    default_style='"',
                )

    egymmlu_subcategories = [f"egymmlu_{category}" for category in ALL_CATEGORIES]

    file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
