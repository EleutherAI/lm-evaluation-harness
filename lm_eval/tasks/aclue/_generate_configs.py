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
    "古文单字多义": "polysemy_resolution",
    "诗词情感分类": "poetry_sentiment_analysis",
    "古汉语命名体识别": "named_entity_recognition",
    "古汉语知识": "basic_ancient_chinese",
    "古诗词上下句预测": "poetry_context_prediction",
    "古文断句": "sentence_segmentation",
    "对联": "couplet_prediction",
    "古诗词曲鉴赏": "poetry_appreciate",
    "国学常识": "ancient_chinese_culture",
    "古音学": "ancient_phonetics",
    "通假字": "homographic_character_resolution",
    "古代文学知识": "ancient_literature",
    "医古文": "ancient_medical",
    "古诗词质量评估": "poetry_quality_assessment",
    "古文阅读理解": "reading_comprehension",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="aclue")
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

    for subject_zh, subject_eng in tqdm(SUBJECTS.items()):
        if args.cot_prompt_path is not None:
            description = cot_file[subject_eng]
        else:
            description = (
                f"以下是关于{subject_zh}的单项选择题，请直接给出正确答案的选项。\n\n"
            )

        yaml_dict = {
            "include": base_yaml_name,
            "task": f"aclue_{args.task_prefix}_{subject_eng}"
            if args.task_prefix != ""
            else f"aclue_{subject_eng}",
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
