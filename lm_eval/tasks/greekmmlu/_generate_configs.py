"""
Generate Greek MMLU task configurations from gmmlu_qa.json dataset.
Creates individual YAML files for each subject and category group files.
Also generates subject+level configs for subjects with educational level splits.
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)


# Subject to category mapping based on the 'group' field in the dataset
SUBJECTS = {
    # Humanities
    "Art": "humanities",
    "Greek History": "humanities",
    "Greek Literature": "humanities",
    "Greek Mythology": "humanities",
    "Greek Traditions": "humanities",
    "Prehistory": "humanities",
    "World History": "humanities",
    "World Religions": "humanities",
    # Social Sciences
    "Accounting": "social_sciences",
    "Economics": "social_sciences",
    "Education": "social_sciences",
    "Geography": "social_sciences",
    "Government and Politics": "social_sciences",
    "Law": "social_sciences",
    "Management": "social_sciences",
    "Modern Greek Language": "social_sciences",
    # STEM
    "Agriculture": "stem",
    "Biology": "stem",
    "Chemistry": "stem",
    "Civil Engineering": "stem",
    "Clinical Knowledge": "stem",
    "Computer Networks & Security": "stem",
    "Computer Science": "stem",
    "Electrical Engineering": "stem",
    "Mathematics": "stem",
    "Medicine": "stem",
    "Physics": "stem",
    # Other
    "Driving Rules": "other",
    "General Knowledge": "other",
    "Maritime_Safety_and_Rescue_Operations": "other",
}

# Subjects WITH level splits - these get subject+level configs instead of plain subject configs
SUBJECTS_WITH_LEVELS = {
    "Agriculture": ["Professional", "University"],
    "Art": ["Professional", "Secondary_School", "University"],
    "Computer Science": ["Professional", "University"],
    "Economics": ["Professional", "University"],
    "Education": ["Professional", "University"],
    "Geography": ["Primary_School", "Secondary_School"],
    "Government and Politics": ["Primary_School", "Secondary_School"],
    "Greek History": ["Primary_School", "Professional", "Secondary_School"],
    "Management": ["Professional", "University"],
    "Medicine": ["Professional", "University"],
    "Modern Greek Language": ["Primary_School", "Secondary_School"],
    "Physics": ["Primary_School", "Professional", "University"],
}

# Subjects WITHOUT level splits - these keep plain subject configs
SUBJECTS_WITHOUT_LEVELS = [
    "Accounting",
    "Biology",
    "Chemistry",
    "Civil Engineering",
    "Clinical Knowledge",
    "Computer Networks & Security",
    "Driving Rules",
    "Electrical Engineering",
    "General Knowledge",
    "Greek Literature",
    "Greek Mythology",
    "Greek Traditions",
    "Law",
    "Mathematics",
    "Prehistory",
    "World History",
    "World Religions",
    "Maritime Safety and Rescue Operations",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="_default_greekmmlu_template_yaml")
    parser.add_argument("--save_prefix_path", default="greekmmlu")
    # parser.add_argument("--data_path", default="/home/mersin-konomi/gmmlu_qa.json")
    return parser.parse_args()


def normalize_subject_name(subject):
    """Convert subject name to valid filename."""
    return subject.lower().replace(" ", "_").replace("&", "and")


if __name__ == "__main__":
    args = parse_args()

    # Get filename of base_yaml so we can `"include":` it in our "other" YAMLs
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]

    eval_logger.info("Generating configs for Greek MMLU...")

    ALL_CATEGORIES = []
    all_tasks = []  # Track all tasks for main benchmark file

    # 1. Generate plain subject configs (for subjects WITHOUT level splits)
    print("\n=== Generating Plain Subject Configs ===")
    for subject in tqdm(SUBJECTS_WITHOUT_LEVELS, desc="Plain subject configs"):
        category = SUBJECTS[subject]
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        normalized_subject = normalize_subject_name(subject)
        hf_config_name = subject.replace(" ", "_").replace("&", "and")

        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"greekmmlu_{category}_tasks",
            "task": f"greekmmlu_{normalized_subject}",
            "task_alias": subject,
            "dataset_name": hf_config_name,
        }

        file_save_path = f"{args.save_prefix_path}_{normalized_subject}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

        all_tasks.append(f"greekmmlu_{normalized_subject}")
        print(f"  ✓ {file_save_path}")

    # 2. Generate subject+level configs (for subjects WITH level splits)
    print("\n=== Generating Subject+Level Configs ===")
    for subject, levels in tqdm(
        SUBJECTS_WITH_LEVELS.items(), desc="Subject+level configs"
    ):
        category = SUBJECTS[subject]
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        normalized_subject = normalize_subject_name(subject)

        for level in levels:
            # Config name: Subject_Level (e.g., Physics_Professional)
            hf_config_name = f"{subject.replace(' ', '_').replace('&', 'and')}_{level}"
            normalized_level = level.lower()
            task_name = f"greekmmlu_{normalized_subject}_{normalized_level}"

            # Alias: Subject_Level e.g. "Physics_Professional"
            task_alias = f"{subject}_{level}"

            yaml_dict = {
                "include": base_yaml_name,
                "tag": f"greekmmlu_{category}_tasks",
                "task": task_name,
                "task_alias": task_alias,
                "dataset_name": hf_config_name,
            }

            file_save_path = (
                f"{args.save_prefix_path}_{normalized_subject}_{normalized_level}.yaml"
            )
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    yaml_dict,
                    yaml_file,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )

            all_tasks.append(task_name)
            print(f"  ✓ {file_save_path}")

    # 3. Generate category group files (using tag as task instead of listing individual tasks)
    print("\n=== Generating Category Group Configs ===")
    for category in ALL_CATEGORIES:
        # Count tasks for this category (for logging purposes)
        task_count = 0
        for subject in SUBJECTS_WITHOUT_LEVELS:
            if SUBJECTS[subject] == category:
                task_count += 1
        for subject, levels in SUBJECTS_WITH_LEVELS.items():
            if SUBJECTS[subject] == category:
                task_count += len(levels)

        file_save_path = f"_greekmmlu_{category}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": f"greekmmlu_{category}",
                    "group_alias": category,
                    "task": [f"greekmmlu_{category}_tasks"],
                    "aggregate_metric_list": [
                        {"metric": "acc", "weight_by_size": True}
                    ],
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
            )
        print(
            f"  ✓ {file_save_path} (task: greekmmlu_{category}_tasks, {task_count} tasks)"
        )

    # 4. Generate main benchmark file
    print("\n=== Generating Main Benchmark Config ===")
    greekmmlu_subcategories = [f"greekmmlu_{category}" for category in ALL_CATEGORIES]

    file_save_path = f"_{args.save_prefix_path}.yaml"
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": "greekmmlu",
                "task": greekmmlu_subcategories,
                "aggregate_metric_list": [
                    {"metric": "acc", "aggregation": "mean", "weight_by_size": True}
                ],
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
    print(f"  ✓ {file_save_path}")

    print("\n✅ Generation complete!")
    print(f"   Plain subject configs: {len(SUBJECTS_WITHOUT_LEVELS)}")
    print(
        f"   Subject+level configs: {sum(len(v) for v in SUBJECTS_WITH_LEVELS.values())}"
    )
    print(f"   Category groups: {len(ALL_CATEGORIES)}")
    print(f"   Total tasks: {len(all_tasks)}")
