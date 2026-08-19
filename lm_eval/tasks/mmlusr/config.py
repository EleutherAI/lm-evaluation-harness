"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)


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

GROUPS = ["answer_only", "question_only", "question_and_answer"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate configuration YAML files for LM Evaluation Harness."
    )
    # Path to the base YAML file from which to inherit settings
    parser.add_argument(
        "--base_yaml_path",
        required=True,
        help="Path to the base YAML configuration file.",
    )

    # Directory where the generated YAML files will be saved
    parser.add_argument(
        "--save_dir",
        default="/data/local/cat/lm-evaluation-harness/lm_eval/tasks/mmlusr/question_and_answer",
    )

    # Optional prefix to add to task names in the YAML files
    parser.add_argument("--task_prefix", default="")

    parser.add_argument("--cot_prompt_path", default=None)

    # Optional prefix to add to group names in the YAML files
    parser.add_argument("--group_prefix", default="")

    # Which MMLU-SR variant to generate. Each one lives in its own directory
    # and reads its own base YAML.
    parser.add_argument("--group", choices=GROUPS, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load base YAML configuration
    # Only the file name is needed, for the "include" key. Parsing the base
    # YAML here used to fail anyway, since it carries a !function tag.
    base_yaml_name = os.path.basename(args.base_yaml_path)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    for group in [args.group] if args.group else GROUPS:
        for subject, category in tqdm(SUBJECTS.items()):
            if args.cot_prompt_path is not None:
                description = cot_file[subject]
            else:
                description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

            yaml_dict = {
                "include": base_yaml_name,
                "tag": f"mmlusr_{args.group_prefix}{group}_{category}"
                if args.group_prefix
                else f"mmlusr_{group}_{category}",
                "task": f"mmlusr_{args.task_prefix}{group}_{subject}"
                if args.task_prefix
                else f"mmlusr_{group}_{subject}",
                "task_alias": subject.replace("_", " "),
                "description": description,
                # The Hub repo no longer exposes a config per subject, so each
                # task reads that subject's CSVs directly. They have no header
                # row, hence the explicit column names.
                "dataset_kwargs": {
                    "data_files": {
                        "train": f"{group}_dev/{group}_{subject}_dev.csv",
                        "test": f"{group}_test/{group}_{subject}_test.csv",
                    },
                    "column_names": [
                        "question",
                        "choice1",
                        "choice2",
                        "choice3",
                        "choice4",
                        "answer",
                    ],
                    "header": None,
                },
            }

            # File path for saving the generated YAML file
            file_save_path = os.path.join(args.save_dir, f"{group}_{subject}.yaml")
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(yaml_dict, yaml_file, allow_unicode=True, default_style='"')
            eval_logger.info(f"Saved YAML for {group} {subject} to {file_save_path}")

    # Save group configuration if specified
    if args.group_prefix:
        file_save_path = os.path.join(
            args.save_prefix_path, args.group_prefix + ".yaml"
        )
        eval_logger.info(f"Saving benchmark config to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(yaml_dict, yaml_file, indent=4, default_flow_style=False)
