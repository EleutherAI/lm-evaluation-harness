"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import os

import pandas as pd
import yaml
from tqdm import tqdm


# Copy from https://github.com/iKala/ievals/blob/main/ievals/settings.py
# from TMMLU+ offical example
categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
    ],
    "humanities": ["history", "philosophy", "law"],
    "social_sciences": [
        "politics",
        "culture",
        "economics",
        "geography",
        "psychology",
        "education",
    ],
    "other": ["other", "business", "health"],  # (business, health, misc.)
}

task_list = [
    "engineering_math",
    "dentistry",
    "traditional_chinese_medicine_clinical_medicine",
    "clinical_psychology",
    "technical",
    "culinary_skills",
    "mechanical",
    "logic_reasoning",
    "real_estate",
    "general_principles_of_law",
    "finance_banking",
    "anti_money_laundering",
    "ttqav2",
    "marketing_management",
    "business_management",
    "organic_chemistry",
    "advance_chemistry",
    "physics",
    "secondary_physics",
    "human_behavior",
    "national_protection",
    "jce_humanities",
    "politic_science",
    "agriculture",
    "official_document_management",
    "financial_analysis",
    "pharmacy",
    "educational_psychology",
    "statistics_and_machine_learning",
    "management_accounting",
    "introduction_to_law",
    "computer_science",
    "veterinary_pathology",
    "accounting",
    "fire_science",
    "optometry",
    "insurance_studies",
    "pharmacology",
    "taxation",
    "education_(profession_level)",
    "economics",
    "veterinary_pharmacology",
    "nautical_science",
    "occupational_therapy_for_psychological_disorders",
    "trust_practice",
    "geography_of_taiwan",
    "physical_education",
    "auditing",
    "administrative_law",
    "basic_medical_science",
    "macroeconomics",
    "trade",
    "chinese_language_and_literature",
    "tve_design",
    "junior_science_exam",
    "junior_math_exam",
    "junior_chinese_exam",
    "junior_social_studies",
    "tve_mathematics",
    "tve_chinese_language",
    "tve_natural_sciences",
    "junior_chemistry",
    "music",
    "education",
    "three_principles_of_people",
    "taiwanese_hokkien",
]
subject2name = {}
# subject2category = {}
SUBJECTS = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="tmmluplus")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    parser.add_argument("--subject_file", default="subject.tsv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from pathlib import Path

    # Initialization
    SUBJECT_FILE = Path(__file__).parent / Path(args.subject_file)

    df = pd.read_csv(SUBJECT_FILE, delimiter="\t")

    for _, row in df.iterrows():
        for _c in categories:
            if row["subject"] in SUBJECTS:
                raise ValueError("Duplicate tasks.")
            if row["category"] in categories[_c]:  # append new item into SUBJECTS
                SUBJECTS[row["subject"]] = _c
                subject2name[row["subject"]] = row["name"]
                break
    # End of SUBJECTS initialization

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path) as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path) as f:
            cot_file = json.load(f)

    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        if args.cot_prompt_path is not None:
            description = cot_file[subject]
        else:
            name_of_subject = subject2name[subject].replace("＿", " ")
            description = f"以下為{name_of_subject}的單選題，請提供正確答案的選項。\n\n"
            # description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "group": f"tmmluplus_{args.task_prefix}_{category}"
            if args.task_prefix != ""
            else f"tmmluplus_{category}",
            "group_alias": category.replace("_", " "),
            "task": f"tmmluplus_{args.task_prefix}_{subject}"
            if args.task_prefix != ""
            else f"tmmluplus_{subject}",
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        # eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                # width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )

    if args.task_prefix != "":
        mmlu_subcategories = [
            f"tmmluplus_{args.task_prefix}_{category}" for category in ALL_CATEGORIES
        ]
    else:
        mmlu_subcategories = [f"tmmluplus_{category}" for category in ALL_CATEGORIES]

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = args.save_prefix_path + ".yaml"

    # eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w") as yaml_file:
        yaml.dump(
            {
                "group": f"tmmluplus_{args.task_prefix}"
                if args.task_prefix != ""
                else "tmmluplus",
                "task": mmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
