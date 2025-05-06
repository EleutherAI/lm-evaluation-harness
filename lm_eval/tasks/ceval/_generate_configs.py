"""
Take in a YAML, and output all other splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)


SUBJECTS = {
    "computer_network": "计算机网络",
    "operating_system": "操作系统",
    "computer_architecture": "计算机组成",
    "college_programming": "大学编程",
    "college_physics": "大学物理",
    "college_chemistry": "大学化学",
    "advanced_mathematics": "高等数学",
    "probability_and_statistics": "概率统计",
    "discrete_mathematics": "离散数学",
    "electrical_engineer": "注册电气工程师",
    "metrology_engineer": "注册计量师",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理",
    "high_school_chemistry": "高中化学",
    "high_school_biology": "高中生物",
    "middle_school_mathematics": "初中数学",
    "middle_school_biology": "初中生物",
    "middle_school_physics": "初中物理",
    "middle_school_chemistry": "初中化学",
    "veterinary_medicine": "兽医学",
    "college_economics": "大学经济学",
    "business_administration": "工商管理",
    "marxism": "马克思主义基本原理",
    "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论体系概论",
    "education_science": "教育学",
    "teacher_qualification": "教师资格",
    "high_school_politics": "高中政治",
    "high_school_geography": "高中地理",
    "middle_school_politics": "初中政治",
    "middle_school_geography": "初中地理",
    "modern_chinese_history": "近代史纲要",
    "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "logic": "逻辑学",
    "law": "法学",
    "chinese_language_and_literature": "中国语言文学",
    "art_studies": "艺术学",
    "professional_tour_guide": "导游资格",
    "legal_professional": "法律职业资格",
    "high_school_chinese": "高中语文",
    "high_school_history": "高中历史",
    "middle_school_history": "初中历史",
    "civil_servant": "公务员",
    "sports_science": "体育学",
    "plant_protection": "植物保护",
    "basic_medicine": "基础医学",
    "clinical_medicine": "临床医学",
    "urban_and_rural_planner": "注册城乡规划师",
    "accountant": "注册会计师",
    "fire_engineer": "注册消防工程师",
    "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "tax_accountant": "税务师",
    "physician": "医师资格",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="ceval-valid")
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

    for subject_eng, subject_zh in tqdm(SUBJECTS.items()):
        if args.cot_prompt_path is not None:
            description = cot_file[subject_eng]
        else:
            description = (
                f"以下是中国关于{subject_zh}的单项选择题，请选出其中的正确答案。\n\n"
            )

        yaml_dict = {
            "include": base_yaml_name,
            "task": f"ceval-valid_{args.task_prefix}_{subject_eng}"
            if args.task_prefix != ""
            else f"ceval-valid_{subject_eng}",
            "dataset_name": subject_eng,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject_eng}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject_eng} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )

    # write group config out

    group_yaml_dict = {
        "group": "ceval-valid",
        "task": [f"ceval-valid_{task_name}" for task_name in SUBJECTS.keys()],
        "aggregate_metric_list": [
            {"metric": "acc", "aggregation": "mean", "weight_by_size": True},
            {"metric": "acc_norm", "aggregation": "mean", "weight_by_size": True},
        ],
        "metadata": {"version": 1.0},
    }

    file_save_path = "_" + args.save_prefix_path + ".yaml"

    with open(file_save_path, "w", encoding="utf-8") as group_yaml_file:
        yaml.dump(
            group_yaml_dict,
            group_yaml_file,
            width=float("inf"),
            allow_unicode=True,
            default_style='"',
        )
