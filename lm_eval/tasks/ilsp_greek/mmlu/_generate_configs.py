"""
Generate the per-subject, per-category, and top-level group YAMLs for the Greek
MMLU (ilsp/mmlu_greek), mirroring lm_eval/tasks/mmlu/default.

ilsp/mmlu_greek keeps the exact field schema of the original MMLU
(`question`, `choices` [list of 4], `answer` [int 0-3]) and the same English
config names, so every subject inherits `_default_template_yaml` unchanged and
only overrides `dataset_name`. The topic-hint `description` is kept in English to
match the base MMLU task and the rest of the ilsp_greek suite (translate the data,
not the template).

Run from this directory:
    python _generate_configs.py
"""

import logging

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)


# Same subject -> category mapping as lm_eval/tasks/mmlu/_generate_configs.py.
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

BASE_YAML_NAME = "_default_template_yaml"
TASK_PREFIX = "mmlu_greek"


if __name__ == "__main__":
    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

        yaml_dict = {
            "include": BASE_YAML_NAME,
            "tag": f"{TASK_PREFIX}_{category}_tasks",
            "task": f"{TASK_PREFIX}_{subject}",
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = f"{TASK_PREFIX}_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
            )

    # One group per category, collecting its subjects via the shared tag.
    for category in ALL_CATEGORIES:
        file_save_path = f"_{TASK_PREFIX}_{category}.yaml"
        eval_logger.info(f"Saving category group to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": f"{TASK_PREFIX}_{category}",
                    "group_alias": category.replace("_", " "),
                    "task": [f"{TASK_PREFIX}_{category}_tasks"],
                    "aggregate_metric_list": [
                        {"metric": "acc", "weight_by_size": True}
                    ],
                    "metadata": {"version": 2},
                },
                yaml_file,
                allow_unicode=True,
                sort_keys=False,
            )

    # Top-level group over the four categories.
    file_save_path = f"_{TASK_PREFIX}.yaml"
    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": TASK_PREFIX,
                "task": [f"{TASK_PREFIX}_{category}" for category in ALL_CATEGORIES],
                "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}],
                "metadata": {"version": 2},
            },
            yaml_file,
            allow_unicode=True,
            sort_keys=False,
        )
