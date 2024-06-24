"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import os

import yaml
from loguru import logger as eval_logger
from tqdm import tqdm


SUBJECTS = {
    "abstract_algebra": ("stem", "абстрактная_алгебра"),
    "anatomy": ("stem", "анатомия"),
    "astronomy": ("stem", "астрономия"),
    "business_ethics": ("other", "этика_бизнеса"),
    "clinical_knowledge": ("other", "клинические_знания"),
    "college_biology": ("stem", "вузовская_биология"),
    "college_chemistry": ("stem", "вузовская_химия"),
    "college_computer_science": ("stem", "вузовская_информатика"),
    "college_mathematics": ("stem", "вузовская_математика"),
    "college_medicine": ("other", "вузовская_медицина"),
    "college_physics": ("stem", "вузовская_физика"),
    "computer_security": ("stem", "компьютерная_безопасность"),
    "conceptual_physics": ("stem", "концептуальная_физика"),
    "econometrics": ("social_sciences", "эконометрика"),
    "electrical_engineering": ("stem", "электротехника"),
    "elementary_mathematics": ("stem", "элементарная_математика"),
    "formal_logic": ("humanities", "формальная_логика"),
    "global_facts": ("other", "глобальные_факты"),
    "high_school_biology": ("stem", "школьная_биология"),
    "high_school_chemistry": ("stem", "школьная_химия"),
    "high_school_computer_science": ("stem", "школьная_информатика"),
    "high_school_european_history": ("humanities", "школьная_европейская_история"),
    "high_school_geography": ("social_sciences", "школьная_география"),
    "high_school_government_and_politics": ("social_sciences", "школьное_государственное_управление_и_политика"),
    "high_school_macroeconomics": ("social_sciences", "школьная_макроэкономика"),
    "high_school_mathematics": ("stem", "школьная_математика"),
    "high_school_microeconomics": ("social_sciences", "школьная_микроэкономика"),
    "high_school_physics": ("stem", "школьная_физика"),
    "high_school_psychology": ("social_sciences", "школьная_психология"),
    "high_school_statistics": ("stem", "школьная_статистика"),
    "high_school_us_history": ("humanities", "школьная_история_США"),
    "high_school_world_history": ("humanities", "школьная_всемирная_история"),
    "human_aging": ("other", "старение_человека"),
    "human_sexuality": ("social_sciences", "человеческая_сексуальность"),
    "international_law": ("humanities", "международное_право"),
    "jurisprudence": ("humanities", "юриспруденция"),
    "logical_fallacies": ("humanities", "логические_ошибки"),
    "machine_learning": ("stem", "машинное_обучение"),
    "management": ("other", "менеджмент"),
    "marketing": ("other", "маркетинг"),
    "medical_genetics": ("other", "медицинская_генетика"),
    "miscellaneous": ("other", "разное"),
    "moral_disputes": ("humanities", "моральные_спор"),
    "moral_scenarios": ("humanities", "моральные_сценарии"),
    "nutrition": ("other", "питание"),
    "philosophy": ("humanities", "философия"),
    "prehistory": ("humanities", "доисторический_период"),
    "professional_accounting": ("other", "профессиональный_учет"),
    "professional_law": ("humanities", "профессиональное_право"),
    "professional_medicine": ("other", "профессиональная_медицина"),
    "professional_psychology": ("social_sciences", "профессиональная_психология"),
    "public_relations": ("social_sciences", "связи_с_общественностью"),
    "security_studies": ("social_sciences", "исследования_в_области_безопасности"),
    "sociology": ("social_sciences", "социология"),
    "us_foreign_policy": ("social_sciences", "внешняя_политика_США"),
    "virology": ("other", "вирусология"),
    "world_religions": ("humanities", "мировые_религии"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="mmlu")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path) as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path) as f:
            cot_file = json.load(f)

    ALL_CATEGORIES = []
    for subject, (category, subject_ru) in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        if args.cot_prompt_path is not None:
            description = cot_file[subject]
        else:
            description = f"Ниже приведены вопросы с несколькими вариантами ответов и одним правильным на тему {subject_ru.replace('_', ' ')}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "group": f"mmlu_{args.task_prefix}_{category}" if args.task_prefix != "" else f"mmlu_{category}",
            "group_alias": category.replace("_", " "),
            "task": f"mmlu_{args.task_prefix}_{subject}" if args.task_prefix != "" else f"mmlu_{subject}",
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                # width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )

    if args.task_prefix != "":
        mmlu_subcategories = [f"mmlu_{args.task_prefix}_{category}" for category in ALL_CATEGORIES]
    else:
        mmlu_subcategories = [f"mmlu_{category}" for category in ALL_CATEGORIES]

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w") as yaml_file:
        yaml.dump(
            {
                "group": f"mmlu_{args.task_prefix}" if args.task_prefix != "" else "mmlu",
                "task": mmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
            width=1000
        )
