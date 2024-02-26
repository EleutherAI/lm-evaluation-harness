"""
Take in a YAML, and output all other splits with this YAML
"""
import argparse
import os

import yaml
from tqdm import tqdm


SUBJECTS = {
    "abstract_algebra": "ألعلوم وتقنية المعلومات و الرياضيات",
    "anatomy": "ألعلوم وتقنية المعلومات و الرياضيات",
    "astronomy": "ألعلوم وتقنية المعلومات و الرياضيات",
    "business_ethics": "علوم أخرى",
    "clinical_knowledge": "علوم أخرى",
    "college_biology": "ألعلوم وتقنية المعلومات و الرياضيات",
    "college_chemistry": "ألعلوم وتقنية المعلومات و الرياضيات",
    "college_computer_science": "ألعلوم وتقنية المعلومات و الرياضيات",
    "college_mathematics": "ألعلوم وتقنية المعلومات و الرياضيات",
    "college_medicine": "علوم أخرى",
    "college_physics": "ألعلوم وتقنية المعلومات و الرياضيات",
    "computer_security": "ألعلوم وتقنية المعلومات و الرياضيات",
    "conceptual_physics": "ألعلوم وتقنية المعلومات و الرياضيات",
    "econometrics": "العلوم الإجتماعية",
    "electrical_engineering": "ألعلوم وتقنية المعلومات و الرياضيات",
    "elementary_mathematics": "ألعلوم وتقنية المعلومات و الرياضيات",
    "formal_logic": "العلوم الانسانية",
    "global_facts": "علوم أخرى",
    "high_school_biology": "ألعلوم وتقنية المعلومات و الرياضيات",
    "high_school_chemistry": "ألعلوم وتقنية المعلومات و الرياضيات",
    "high_school_computer_science": "ألعلوم وتقنية المعلومات و الرياضيات",
    "high_school_european_history": "العلوم الانسانية",
    "high_school_geography": "العلوم الإجتماعية",
    "high_school_government_and_politics": "العلوم الإجتماعية",
    "high_school_macroeconomics": "العلوم الإجتماعية",
    "high_school_mathematics": "ألعلوم وتقنية المعلومات و الرياضيات",
    "high_school_microeconomics": "العلوم الإجتماعية",
    "high_school_physics": "ألعلوم وتقنية المعلومات و الرياضيات",
    "high_school_psychology": "العلوم الإجتماعية",
    "high_school_statistics": "ألعلوم وتقنية المعلومات و الرياضيات",
    "high_school_us_history": "العلوم الانسانية",
    "high_school_world_history": "العلوم الانسانية",
    "human_aging": "علوم أخرى",
    "human_sexuality": "العلوم الإجتماعية",
    "international_law": "العلوم الانسانية",
    "jurisprudence": "العلوم الانسانية",
    "logical_fallacies": "العلوم الانسانية",
    "machine_learning": "ألعلوم وتقنية المعلومات و الرياضيات",
    "management": "علوم أخرى",
    "marketing": "علوم أخرى",
    "medical_genetics": "علوم أخرى",
    "miscellaneous": "علوم أخرى",
    "moral_disputes": "العلوم الانسانية",
    "moral_scenarios": "العلوم الانسانية",
    "nutrition": "علوم أخرى",
    "philosophy": "العلوم الانسانية",
    "prehistory": "العلوم الانسانية",
    "professional_accounting": "علوم أخرى",
    "professional_law": "العلوم الانسانية",
    "professional_medicine": "علوم أخرى",
    "professional_psychology": "العلوم الإجتماعية",
    "public_relations": "العلوم الإجتماعية",
    "security_studies": "العلوم الإجتماعية",
    "sociology": "العلوم الإجتماعية",
    "us_foreign_policy": "العلوم الإجتماعية",
    "virology": "علوم أخرى",
    "world_religions": "العلوم الانسانية",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="ammlu")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    for subject_eng, category in tqdm(SUBJECTS.items()):
        if args.cot_prompt_path is not None:
            description = cot_file[subject_eng]
        else:
            description = f"فم بعملية التقييم في مجال {category} \n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "task": f"ammlu_{args.task_prefix}_{subject_eng}"
            if args.task_prefix != ""
            else f"ammlu_{subject_eng}",
            "dataset_name": subject_eng,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject_eng}.yaml"
        print(f"Saving yaml for subset {subject_eng} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )
