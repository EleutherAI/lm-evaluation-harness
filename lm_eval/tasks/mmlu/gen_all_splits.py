"""
Take in a YAML, and output all other splits with this YAML
"""
import os
import yaml
import argparse

from tqdm import tqdm

from lm_eval import utils
from lm_eval.logger import eval_logger

SUBJECTS = [
    # "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--benchmark_name", required=True)
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument(
        "--task_save_path", default="lm_eval/tasks/mmlu/hendrycks_test_original"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path) as f:
        base_yaml = yaml.full_load(f)
    print(base_yaml)

    for subject in tqdm(SUBJECTS):

        yaml_dict = {
            "include": base_yaml_name,
            "task": base_yaml["task"].strip("abstract_algebra") + "subject",
            "dataset_name": subject,
        }

        file_save_path = args.task_save_path + f"_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w") as yaml_file:
            yaml.dump(yaml_dict, yaml_file)
