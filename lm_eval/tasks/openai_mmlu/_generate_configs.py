"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os
from pathlib import Path

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger("lm-eval")


SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

LANGUAGES = {
    "AR_XY": "Arabic (Generic)",
    "BN_BD": "Bengali (Bangladesh)",
    "DE_DE": "German (Germany)",
    "ES_LA": "Spanish (Latin America)",
    "FR_FR": "French (France)",
    "HI_IN": "Hindi (India)",
    "ID_ID": "Indonesian (Indonesia)",
    "IT_IT": "Italian (Italy)",
    "JA_JP": "Japanese (Japan)",
    "KO_KR": "Korean (South Korea)",
    "PT_BR": "Portuguese (Brazil)",
    "ZH_CN": "Chinese (China)",
    "SW_KE": "Swahili (Kenya)",
    "YO_NG": "Yoruba (Nigeria)",
    "EN_US": "English (United States)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="openai_mmlu")
    parser.add_argument("--group_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    ALL_CATEGORIES = []
    for langgode, language_full_name in tqdm(LANGUAGES.items()):
        _langgode = langgode.lower()
        out_folder = Path(_langgode)
        out_folder.mkdir(exist_ok=True)
        for subject, category in SUBJECTS.items():
            if category not in ALL_CATEGORIES:
                ALL_CATEGORIES.append(category)

            description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))} in the {language_full_name} language.\n\n"

            yaml_dict = {
                "include": f"../{base_yaml_name}",
                "tag": f"mmlu_{_langgode}_{category}",
                "task": f"mmlu_{_langgode}_{subject}",
                "task_alias": f'{_langgode} {subject.replace("_", " ")}',
                "dataset_name": subject,
                "test_split": langgode,
                "description": description,
            }

            file_save_path = out_folder / (args.save_prefix_path + f"_{subject}.yaml")
            eval_logger.info(
                f"Saving yaml for subset {_langgode},{subject} to {file_save_path}"
            )
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    yaml_dict,
                    yaml_file,
                    allow_unicode=True,
                    default_style='"',
                )

            # shutil.copy("_default_template.yaml", out_folder/"_default_template.yaml")

        file_save_path = out_folder / (
            "_" + args.save_prefix_path + f"_{_langgode}.yaml"
        )
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            dct = {
                "group": f"openai_mmlu_{_langgode}",
                "group_alias": _langgode,
                "task": [f"mmlu_{_langgode}_tasks"],
                "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}],
                "metadata": {"version": "1.0.0"},
            }

            yaml.dump(
                dct,
                yaml_file,
                indent=4,
                default_flow_style=False,
            )
